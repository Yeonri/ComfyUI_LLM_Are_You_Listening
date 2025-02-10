import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import folder_paths

GLOBAL_Instruct_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM_instruct_models")
GLOBAL_GGUF_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM_gguf_models")

LOADED_MODELS = {}
SUMMARY_STORY = ""
PREVIOUS_PROMPT = ""
DESCRIPTION = ""


STORY_PROMPT = (
    "You are an AI assistant specialized in summarizing stories summarized in English and Korean stories currently entered. "
    "Your mission is to fully grasp the context of the summarized story and the Korean story currently entered, and then update the summarized English story. "
    "1. Mandatory Format: You must print only the summary story. "
    "2. Summarization of Inputs: - Continuously build a coherent summary of the story based on all user inputs up to the current point. "
    "   - Ensure that each subject appears only once in the [Subject] component, eliminating any duplicates. "
    "3. Missing or Partial Information: - If the user’s input lacks certain components (e.g., no explicit Time), use context from the summarized story to maintain a natural, cohesive narrative. "
    "4. Strict Limitations: - Do not introduce new story elements beyond what the user provides. "
    "   - Do not modify or extend the user’s narrative beyond summarization. "
    "   - Use only the content explicitly stated by the user, plus any relevant context from the summarized story to fill in missing details. "
    "# Important: - Add new information without ever deleting existing information. "
    "- Type of Input: Summarized story, current user input. "
    "- Output format: a summary of the summarized story and the current user's input."
)

IMAGE_PROMPT = (
    "You are an AI assistant specialized in converting Korean sentences into English image prompts for image generation models. "
    "Change the contents entered in Korean to English and follow the procedure below. "
    "1. Story contextualization: Understand the overall story of the story by grasping the context of the summarized story and the current story. "
    "2. Contextualize previous image prompts and summary stories: Identify the relationship between the summarized story and the previous image prompt. "
    "3. Create current story image prompts: Use the context of the summarized story you have identified and the previous image prompt context to create a very natural image prompt for the current story. "
    "   You need to accurately grasp the context of the background, object, condition, costume, etc. "
    "# Important: You must only print out image prompts. There should never be any arbitrarily decorated or added content other than the summarized story and the entered story."
)

DESCRIPTION_PROMPT = (
    "You are an AI helper who writes a comprehensive and up-to-date description of the current story state.\n\n"
    "1. You have access to:\n"
    "- The summarized story (which shows how the story has progressed).\n"
    "- The previous description (the prior state of all objects and environment).\n"
    "- The previous image prompt.\n"
    "- The latest user input (new events or interactions).\n\n"
    "2. Your goal is:\n"
    "- To update the **scene description** so that it accurately reflects any new changes from the latest input.\n"
    "- Capture changes in objects, their interactions, or the background.\n"
    "- Ensure consistency with the summarized story so far.\n\n"
    "3. **Important**:\n"
    "- Only output the final updated description.\n"
    "- Do NOT restate the entire story.\n"
    "- Do NOT write any extra explanations or text.\n"
    "- Do NOT include the summary story text itself.\n"
    "- If something changed, update or remove the old information. If new info was introduced, add it.\n\n"
    "4. **Output Format** (strict):\n"
    "- <object>: [features], [emotions], [status], [action/interaction]\n"
    "- <object2>: [features], [emotions], [status], [action/interaction]\n"
    "- <Time/Background>: [Time], [Weather], [Background]\n\n"
    "### Your final answer must strictly follow the bullet format above.\n"
    "### Provide no additional lines, no summary, no repeated context.\n"
)

class BaseAYL_Node:
    def summary_story(self, story, user_input: str):
        combined_input = f"Story_summary: {story}\nUser_Input: {user_input}"
        messages = [
            {"role": "system", "content": STORY_PROMPT},
            {"role": "user", "content": combined_input}
        ]
        return self.generate_model_output(messages)

    def generate_description(self, story, description, user_input: str) -> str:
        combined_input = f"Story summary: {story}\nDescription: {description}\nCurrent story: {user_input}"
        messages = [
            {"role": "system", "content": DESCRIPTION_PROMPT},
            {"role": "user", "content": combined_input}
        ]
        return self.generate_model_output(messages)

    def generate_image_prompt(self, story, description, pre_prompt, user_input: str):
        combined_input = (
            f"Story summary: {story}\n"
            f"Description: {description}\n"
            f"Previous Image Prompt: {pre_prompt}\n"
            f"Current story: {user_input}"
        )
        messages = [
            {"role": "system", "content": IMAGE_PROMPT},
            {"role": "user", "content": combined_input}
        ]
        return self.generate_model_output(messages)

    def generate_model_output(self, messages):
        raise NotImplementedError("Subclass load ERROR")

class AYL_Node(BaseAYL_Node):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = []
        if os.path.isdir(GLOBAL_Instruct_MODEL_DIR):
            model_dirs = [
                d for d in os.listdir(GLOBAL_Instruct_MODEL_DIR)
                if os.path.isdir(os.path.join(GLOBAL_Instruct_MODEL_DIR, d))
            ]
            model_list.extend(model_dirs)
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
                "model": (model_list, ),
                "max_tokens": ("INT", {"default": 4096, "min": 10, "max": 8192}),
            },
            "optional": {
                "ayl_api_node": ("AYL_API", ),
            },
        }
    
    CATEGORY = "Yeonri/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_prompt", "summary_story", "description")

    def main(self, text, model, max_tokens, ayl_api_node=None):
        global LOADED_MODELS, SUMMARY_STORY, PREVIOUS_PROMPT, DESCRIPTION
        model_path = os.path.join(GLOBAL_Instruct_MODEL_DIR, model)
        
        if not LOADED_MODELS:
            if os.path.isdir(model_path):
                snapshot_dir = os.path.join(model_path, "snapshots")
                if os.path.isdir(snapshot_dir):
                    subfolders = [
                        os.path.join(snapshot_dir, folder)
                        for folder in os.listdir(snapshot_dir)
                        if os.path.isdir(os.path.join(snapshot_dir, folder))
                    ]
                    if subfolders:
                        latest_snapshot = max(subfolders, key=os.path.getmtime)
                        print("Most recently received snapshot folder:", latest_snapshot)
                    else:
                        latest_snapshot = None
                        print("There are no subfolders within the snapshots folder.")
                else:
                    latest_snapshot = None
                    print("The snapshot directory does not exist:", snapshot_dir)
                if latest_snapshot is None:
                    raise ValueError("유효한 snapshot folder를 찾을 수 없습니다.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    latest_snapshot,
                    torch_dtype="auto",
                    device_map="auto",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(latest_snapshot)
                LOADED_MODELS[model] = (self.model, self.tokenizer)
            else:
                raise ValueError(f"Model path not found: {model_path}")
        # else:
            # self.model, self.tokenizer = LOADED_MODELS[model]

        self.max_tokens = max_tokens

        if ayl_api_node:
            PREVIOUS_PROMPT = ayl_api_node[0]
            SUMMARY_STORY = ayl_api_node[1]
            DESCRIPTION = ayl_api_node[2]

        SUMMARY_STORY = self.summary_story(SUMMARY_STORY, text)
        DESCRIPTION = self.generate_description(DESCRIPTION, DESCRIPTION, text)
        PREVIOUS_PROMPT = self.generate_image_prompt(PREVIOUS_PROMPT, DESCRIPTION, PREVIOUS_PROMPT, text)
            
        return (PREVIOUS_PROMPT, SUMMARY_STORY, DESCRIPTION)

    def generate_model_output(self, messages):
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class AYL_GGUF_Node(BaseAYL_Node):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = []
        if os.path.isdir(GLOBAL_GGUF_MODEL_DIR):
            gguf_files = [
                file for file in os.listdir(GLOBAL_GGUF_MODEL_DIR)
                if file.endswith('.gguf')
            ]
            model_list.extend(gguf_files)
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
                "model": (model_list, ),
                "max_tokens": ("INT", {"default": 4096, "min": 10, "max": 8192}),
                "n_gpu_layers": ("INT", {"default": -1, "max": 1000}),
                "n_threads": ("INT", {"default": 8, "max": 50}),
            },
            "optional": {
                "ayl_api_node": ("AYL_API", ),
            },
        }
    
    CATEGORY = "Yeonri/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_prompt", "summary_story", "description")

    def main(self, text, model, max_tokens, n_gpu_layers, n_threads, ayl_api_node=None):
        global LOADED_MODELS, SUMMARY_STORY, PREVIOUS_PROMPT, DESCRIPTION
        model_path = os.path.join(GLOBAL_GGUF_MODEL_DIR, model)
        if not LOADED_MODELS:
            if model.endswith(".gguf"):
                self.model = Llama(model_path=model_path, n_ctx=max_tokens, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
                self.max_tokens = max_tokens
                LOADED_MODELS[model] = self.model
            else:
                raise ValueError(f"Invalid GGUF model file: {model}")
        # else:
            # self.model = LOADED_MODELS[model]

        if ayl_api_node:
            PREVIOUS_PROMPT = ayl_api_node[0]
            SUMMARY_STORY = ayl_api_node[1]
            DESCRIPTION = ayl_api_node[2]
        
        SUMMARY_STORY = self.summary_story(SUMMARY_STORY, text)
        DESCRIPTION = self.generate_description(DESCRIPTION, DESCRIPTION, text)
        PREVIOUS_PROMPT = self.generate_image_prompt(PREVIOUS_PROMPT, DESCRIPTION, PREVIOUS_PROMPT, text)

        return (PREVIOUS_PROMPT, SUMMARY_STORY, DESCRIPTION)

    def generate_model_output(self, messages):
        gguf_response = self.model.create_chat_completion(messages=messages)
        return gguf_response["choices"][0]["message"]['content']

class AYL_API_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_prompt": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
                "summary_story": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
                "description": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
            
        }
    
    CATEGORY = "Yeonri/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("AYL_API", )
    RETURN_NAMES = ("ayl_api_node", )
    OUTPUT_NODE = True

    def main(self, previous_prompt="", summary_story="", description="", unique_id=None, extra_pnginfo=None):
        ayl_api_node = (
            previous_prompt,
            summary_story,
            description,
        )
        return (ayl_api_node,)


        
NODE_CLASS_MAPPINGS = {
    "AYL_Node": AYL_Node,
    "AYL_GGUF_Node": AYL_GGUF_Node,
    "AYL_API_Node": AYL_API_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AYL_Node": "AYL_Node",
    "AYL_GGUF_Node": "AYL_GGUF_Node",
    "AYL_API_Node": "AYL_API_Node",
}

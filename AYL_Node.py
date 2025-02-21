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
    """
    You are an AI that generates image prompts for the current story.
    **All contents are translated into English, processed, and printed in English.**
    **Never use description data that you don't need in your current story**
    If the given data is empty, it is the beginning of the story. Create an image prompt by only generating basic information about the current story. Do not create objects other than those specified in the current story.
    Follow the steps below.

    # IMAGE PROMPT FORM:
        "[Targets] [Attire] [Description and Action] [Positioning], [Background], [Time],"
    
    1. Thought
        - Figure out what scene the **current story** depicts.
        - Figure out how the background of the current story should be structured.
        - The objects in the current story determine who exists
        - Determine which part of the background exists for each object
        - **Determine specifically what the objects in the current story are doing**
    
    2. inspection
        - determine what the current story is trying to describe and examine whether the generated image prompts fit your needs.
    
    3. Create an image prompt
        - Create an image prompt based on what you thought in step 1 and 2
        - The scene of the story must be created in detail with the action, background, and effect of the object.
        - Please specify the exact location of each object in order to clearly display the object in the image.
        - **Describe the behavior and interaction of the object in more detail.**

    4. improvement
        - **Please think of an improved image prompt so that the generated image prompt is dramatically represented for the story before generating it**
        - **Describe the behavior and interaction of the object in more detail.**
        - **Please express the emotions and facial expressions of the object in detail**
        - **Create an object's attribute [age, hair color, cloning style, color scheme, basic body shape]**
        - Don't add anyone you don't need in your current story.

    5. Staying Consistent
        - Access the description in which the contents of the previous story are saved.
        - Do an analysis of the previous story and the current story.
    
        - Identify if there is a change in the object in the story. 
            - If a change exists, it ignores the information of the object written in the description, because the description contains previous information.
            - If no change exists, reconstruct the object's appearance with the information in the description from the image prompt you created, because it needs to be consistent.

        - Identify if there is a change in the background in the story.
            - If the change exists, ignore the background written in the description, as it is previous information.
            - If the change doesn't exist, then change the background of the image prompt to the background written in the description, because it needs to be consistent.
    
    6. Analyze the image prompts.
        - Is the content of the image prompt focused on the current story? Explain the rationale in detail.
        - Is the image prompt focused on the behavior of each object in the current story? Explain the rationale in detail.
        - If the main object's actions exist in the story, does the image prompt appropriately reflect the emotional and behavioral changes of each object due to those actions? Explain the rationale in detail.
        - If not, please find out the detailed reasons and explanations for 1 and 2 and 3 reconstruct the image prompt to focus on the current story.
    
    7. Final Image Prompt
    - **Create a final image prompt that aggregates everything**
    - Begin with: "best quality,"
    
    """
)

DESCRIPTION_PROMPT = (
    """
    You are an AI that analyzes image prompts and stores each object and background information about the current story.
    **All contents are translated into English, processed, and printed in English.**
    
    1. Analyze the image prompts based on the current story.
    2. Identify the information and background information of each object.
    3. Use the results of 1 and 2 to simply write DESCRIPTION FORM to keep the image prompt consistent.
    4. If an object disappears, do not delete it; instead, write "Noted in the present story" in [STATUS] as it may be used in later stories.
    5. **The TARGET object only stores key characters (never add buildings, objects, etc.)**
    
    # DESCRIPTION FORM
    [TARGET] (**Major characters only**)
      - [ATTIRE] (**Do not include an object's behavior.** e.g., age, hair color, clothing style, color scheme, weapons/items, basic body shape)
      - [EMOTION] (e.g., angry, happy, scared, fearful)
      - [STATUS] (e.g., dead or Not needed in the present story or Existence)

    [BACKGROUND] (Use only one-word or simple keywords. Do not add any extra details, e.g., desert, village, kingdom)

    **Output only the DESCRIPTION FORM as specified, with no additional text or explanation.**
    **Never add data other than DESCRIPTION FROM.**
    """
)

class BaseAYL_Node:
    def summary_story(self, story, user_input: str):
        combined_input = f"Story_summary: {story}\nUser_Input: {user_input}"
        messages = [
            {"role": "system", "content": STORY_PROMPT},
            {"role": "user", "content": combined_input}
        ]
        return self.generate_model_output(messages)

    def generate_description(self, story, description, previous_pronpt, user_input: str) -> str:
        combined_input = (
            # f"[STORY SUMMARY START] {self.summary_story_text} [STORY SUMMARY END]\n\n"
            f"[CURRENT STORY START] {user_input} [CURRENT STORY END]\n\n"
            # f"[PREVIOUS DESCRIPTION START] {self.description} [PREVIOUS DESCRIPTION START]"
            f"Previous Image Prompt: {previous_pronpt}"
        )
        messages = [
            {"role": "system", "content": DESCRIPTION_PROMPT},
            {"role": "user", "content": combined_input}
        ]
        return self.generate_model_output(messages)

    def generate_image_prompt(self, story, description, pre_prompt, user_input: str):
        combined_input = (
            # f"[STORY SUMMARY START] {self.summary_story_text} [STORY SUMMARY END]\n\n"
            # f"Previous prompt: {self.previous_prompt}\n"
            f"[DESCRIPTION START] {description} [DESCRIPTION END]\n\n"
            f"[Current STORY START] {user_input} [CURRENT STORY END]"
        )   
        messages = [
                {"role": "system",
                "content": IMAGE_PROMPT
                },
                {"role": "user",
                "content": combined_input}
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
        PREVIOUS_PROMPT = self.generate_image_prompt(PREVIOUS_PROMPT, DESCRIPTION, PREVIOUS_PROMPT, text)
        DESCRIPTION = self.generate_description(DESCRIPTION, DESCRIPTION, PREVIOUS_PROMPT, text)
          
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

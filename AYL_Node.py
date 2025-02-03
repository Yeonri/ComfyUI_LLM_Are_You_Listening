import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths

GLOBAL_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM_models")

ASSETS_DIR = "./web/assets/js"

LOADED_MODELS = {}
SUMMARY_STORY = ""
PREVIOUS_PROMPT = ""


STORY_PROMPT = (
    "You are an AI assistant specialized in summarizing stories summarized in English and Korean stories currently entered."
    "Your mission is to fully grasp the context of the summarized story and the Korean story currently entered, and then update the summarized English story."

    "1. Mandatory Format"
        "You must print only the summary story."

    "2. Summarization of Inputs"
    "- Continuously build a coherent summary of the story based on all user inputs up to the current point."
    "- Ensure that each subject appears only once in the [Subject] component, eliminating any duplicates."

    "3. Missing or Partial Information"
    "- If the user’s input lacks certain components (e.g., no explicit Time), use context from the summarized story to maintain a natural, cohesive narrative."

    "4. Strict Limitations"
    "- Do not introduce new story elements beyond what the user provides."
    "- Do not modify or extend the user’s narrative beyond summarization."
    "- Use only the content explicitly stated by the user, plus any relevant context from the summarized story to fill in missing details."

    "# Important"
    "- Add new information without ever deleting existing information."
    "- Type of Input: Summarized story, current user input"
    "- Output format: a summary of the summarized story and the current user's input."
)


IMAGE_PROMPT = (
    "You are an AI assistant specialized in converting Korean sentences into English image prompts for image generation models."
    "Change the contents entered in Korean to English and follow the procedure below."

    "1. Story contextualization"
    "Understand the overall story of the story by grasping the context of the summarized story and the current story."

    "2. Contextualize previous image prompts and summary stories"
    "Identify the relationship between the summarized story and the previous image prompt."

    "3. Create current story image prompts"
    "Use the context of the summarized story you have identified and the previous image prompt context to create a very natural image prompt for the current story."
    "You need to accurately grasp the context of the background, object, condition, costume, etc."

    "# Important"
    "You must only print out image prompts."
    "There should never be any arbitrarily decorated or added content other than the summarized story and the entered story."
)


class AnyType(str):
    
    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")

class AYL_Node:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = []

        if os.path.isdir(GLOBAL_MODEL_DIR):
            model_dirs = [d for d in os.listdir(GLOBAL_MODEL_DIR) 
                          if os.path.isdir(os.path.join(GLOBAL_MODEL_DIR, d))]
            model_list.extend(model_dirs)



        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompt": True, "default": ""}),
                "model": (model_list, ),
                "max_tokens": ("INT", {"default": 4096, "min": 10, "max": 8192}),
            }
        }
    
    CATEGORY = "Yeonri/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("generated", "Text Prompt", "Summary Text")


    def __init__(self):
        global LOADED_MODELS

    def main(self, text, model, max_tokens):
        global LOADED_MODELS, SUMMARY_STORY, PREVIOUS_PROMPT

        model_path = os.path.join(GLOBAL_MODEL_DIR, model)

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
                        latest_sanpshot = max(subfolders, key=os.path.getmtime)
                        print("Most recently received snapshot folder:", latest_sanpshot)
                    else:
                        print("There are no subfolders within the snapshots folder.")
                else:
                    print("The snapshot directory does not exist:", snapshot_dir)

                model_name = latest_sanpshot

                self.max_tokens = max_tokens

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                LOADED_MODELS[model] = (self.model, self.tokenizer)

        SUMMARY_STORY = self.summary_story(SUMMARY_STORY, text)
        PREVIOUS_PROMPT = self.generate_prompt(SUMMARY_STORY, PREVIOUS_PROMPT, text)

        return (SUMMARY_STORY, PREVIOUS_PROMPT, SUMMARY_STORY)

    def generate_model_output(self, messages):

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens ##
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def summary_story(self, story, user_input: str):
        
        combined_input = f"Story_summary: {story}\nUser_Input: {user_input}"

        messages = [
                {"role": "system",
                "content": STORY_PROMPT},
                {"role": "user",
                "content": combined_input}
            ]


        return self.generate_model_output(messages)
        

    def generate_prompt(self, story, pre_prompt, user_input: str):
            
        user_input = f"Story_summary: {story}\nPrevious prompt: {pre_prompt}\nCurrent_story{user_input}"
        
        messages = [
                    {"role": "system",
                    "content": IMAGE_PROMPT
                    },
                    {"role": "user",
                    "content":
                        f"Story summary: A cosmic princess discovers an interstellar garden.\n"
                        f"Previous prompt: A mesmerizing, futuristic, photorealistic portrait captures a cosmic princess in an interstellar garden. Seated gracefully on a hovering crystalline seat, she wears a flowing, iridescent gown that shimmers with starlight. Her long, lavender hair floats gently in the zero-gravity environment, and her tranquil expression enhances the ethereal ambiance. The garden features bioluminescent alien flora of vibrant hues, swirling nebula clouds visible through a transparent dome, and subtle hovering lights that glitter like fireflies. High-definition details highlight her luminous skin, delicate cosmic makeup, and the complex designs woven into her gown. The scene glows with soft cosmic light, intensifying the dreamy, otherworldly setting.\n"
                        f"Current_stroy: A cosmic princess sits on a hovering crystalline seat in an interstellar garden."},
                    {"role": "assistant",
                    "content":
                        f"In a striking, photorealistic scene, a cosmic princess resides amid an interstellar garden brimming with bioluminescent alien plants. She wears a flowing, iridescent gown that catches the faint starlight filtering through a transparent dome overhead. Her lavender hair seems to drift in a gentle, zero-gravity setting, and her tranquil gaze conveys both wisdom and curiosity. The surrounding foliage pulsates with neon hues, while delicate orbs of light hover nearby like fireflies. Subtle cosmic clouds swirl beyond the dome, hinting at the vast expanse of space. The image emphasizes high-definition details, revealing the princess’s radiant skin, intricate patterns within her gown, and the shimmering texture of her crystalline seat, culminating in a breathtaking fusion of futuristic fantasy and celestial beauty."},
                    {"role": "user",
                    "content":
                        f"Story summary: The cosmic princess remains in her interstellar garden, but an unexpected cosmic storm begins to surge around her.\n"
                        f"Previous prompt: A cosmic princess sits on a hovering crystalline seat in an interstellar garden. She wears a flowing, iridescent gown and has lavender hair drifting in zero-gravity. The garden is alive with bioluminescent alien flora, neon-hued foliage, and a transparent dome revealing swirling nebula clouds overhead. High-definition details emphasize the princess’s radiant skin, intricate gown patterns, and the crystalline seat’s shimmering texture.\n"
                        f"Current_story: The cosmic princess faces a sudden cosmic storm in her garden."},
                    {"role": "assistant",
                    "content":
                        f"In a striking, photorealistic scene, the cosmic princess stands amid her interstellar garden as a swirling cosmic storm rages around her. Her iridescent gown and lavender hair ripple in the powerful winds, while she holds aloft an ornate staff shimmering with faint starlight. Nearby, a hovering robotic caretaker extends a crackling energy shield, its metallic frame reflecting flashes of cosmic lightning. The garden’s bioluminescent flora glows in vivid neon tones, bending under the storm’s turbulence. Beyond the translucent dome, vibrant nebulas swirl in the distance, casting ethereal glimmers onto the princess’s crystalline seat and the caretaker’s polished surface. The composition emphasizes hyper-detailed textures—from the princess’s gleaming gown to the caretaker’s mechanical joints—creating a dramatic, futuristic atmosphere filled with both wonder and tension."},
                    {"role": "user",
                    "content": user_input}
                ]

        return self.generate_model_output(messages)

NODE_CLASS_MAPPINGS = {
    "AYL_Node": AYL_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AYL_Node": "AYL_Node",
}

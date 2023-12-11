from openai import OpenAI

class CaptionGenerator:
    def __init__(self, api_key, model_id):
        self.api_key = api_key
        self.model_id = model_id

        self.client = OpenAI(api_key = self.api_key) 

        self.prompt_base = "I am creating a dataset for a prompt based music generation system. Generate a prompt that can be used as the label for a music snippet that has the following features:\n"

    def set_prompt_base(self, prompt_base):
        self.prompt_base = prompt_base
        if self.prompt_base[-1] != '\n':
            self.prompt_base += '\n'

    def create_prompt(self, keywords):
        prompt = self.prompt_base
        prompt += str(keywords)

        return prompt

    def generate_caption(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_id,
        )
        caption = response.choices[0].message.content
        return caption

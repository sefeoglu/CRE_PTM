"""Generates responses for prompts using a language model defined in Hugging Face."""
"""Created by: Sefika"""

import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import read_json, write_json
import configparser

class LLM(object):

    def __init__(self, model_id="google/flan-t5-base"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-base".
        """


        self.model, self.tokenizer = self.get_model(model_id)


    def get_model(self, model_id="google/flan-t5-base"):
        """
        Args:
            model_id (str, optional): LLM name at HuggingFace . Defaults to "google/flan-t5-xl".
        Returns:
            model: model from Hugging Face
            tokenizer: tokenizer of this model
        """
        tokenizer = T5Tokenizer.from_pretrained(model_id)

        model = T5ForConditionalGeneration.from_pretrained(model_id,
                                                    device_map="auto",
                                                    load_in_8bit=False,
                                                    torch_dtype=torch.float16)
        return model,tokenizer

    def get_prediction(self, prompt, length=30):
        """
        Args:
            model : loaded model
            tokenizer: loaded tokenizer
            prompt (str): prompt to generate response
            length (int, optional): Response length. Defaults to 30.

        Returns:
            response (str): response from the model
        """

        inputs = self.tokenizer(prompt, add_special_tokens=True, max_length=526,return_tensors="pt").input_ids.to("cuda")

        outputs = self.model.generate(inputs, max_new_tokens=length)

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return response

def main(input_path, output_path):
  llm_instance = LLM()
  data = read_json(input_path)
  responses, relations = [], []

  for i, item in enumerate(data):
    prompt = item['prompt']
    relations.append(item['relation'])

    response = llm_instance.get_prediction(prompt)
    print('test:', i)
    if len(response) == 0:
      print("No response")
      responses.append({"predict":"", "ground_truth": item['relation']})
    else:
      print(response[0])
      responses.append({"predict":response[0], "ground_truth": item['relation']})

  write_json(responses, output_path)
  
if __name__ =="__main__":
   configparser_path = "config.ini"
   config = configparser.ConfigParser()
   config.read(configparser_path)
   input_path = config['ZEROSHOT']['input_path']
   output_path = config['ZEROSHOT']['output_path']
   number_of_tasks = int(config['ZEROSHOT']['number_of_tasks'])
   run_number = int(config['ZEROSHOT']['run_number'])

   for j in range(1, run_number+1):
      for i in range(1, number_of_tasks+1):
        output_path = f"{output_path}/result_task_{j}_{i}.json"
        input_path = f"{input_path}/run_{j}/task{i}/test_1.json"
        main(input_path, output_path)


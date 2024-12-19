"""Generates responses for prompts using a language model defined in Hugging Face."""
"""Created by: Sefika"""

import json
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from transformers import T5Tokenizer, T5ForConditionalGeneration

class LLM(object):

    def __init__(self, model_id="google/flan-t5-base"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-base".
        """
        self.maxmem={i:f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range(4)}
        self.maxmem['cpu']='300GB'
        if model_id=="google/flan-t5-base":
            self.model, self.tokenizer = self.get_model(model_id)
        else:
            self.model, self.tokenizer = self.get_model_decoder(model_id)


    def get_model(self, model_id="google/flan-t5-base"):
        """_summary_

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
        """_summary_

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

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(input_path, output_path):
  llm_instance = LLM()
  tokenizer = llm_instance.tokenizer
  model = llm_instance.model
  data = read_json(input_path)
  responses = []
  relations = []
  for i, item in enumerate(data):
    # print(item)
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
  ## TODO ##
  ## : Change the paths ##
  for j in range(1, 6):
    for i in range(1, 11):
        output_path = f"drive/MyDrive/ESWC-figures/fewrel/zero-shot/result_task_{j}_{i}.json"
        input_path = f"t5_format/test/run_{j}/task{i}/test_1.json"
        main(input_path, output_path)


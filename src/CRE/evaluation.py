
import json


def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_prediction(model,tokenizer, prompt, length=250,stype='greedy'):

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096,return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs,  pad_token_id=tokenizer.eos_token_id, max_new_tokens=length)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return response


def evaluate_model(experiment_id, task_id, model, tokenizer, current_task=True):
    if current_task:
      input_path = "llama_format_data/test/run_{0}/{1}/test_1.json".format(experiment_id, task_id)
      data = read_json(input_path)
      out_pred_path = "drive/MyDrive/ESWC-figures/fewrel/KMmeans_CRE_fewrel_{0}/task_{1}_current_task_pred.json".format(experiment_id, task_id)
      out_acc_path = "drive/MyDrive/ESWC-figures/fewrel/KMmeans_CRE_fewrel_{0}/task_{1}_current_task_result.json".format(experiment_id, task_id)
    else:
      data = []
      for t in range(1, task_id+1):
          input_path = "llama_format_data/test/run_{0}/task{1}/test_1.json".format(experiment_id, t)
          task_data = read_json(input_path)
          data.extend(task_data)
      out_pred_path = "drive/MyDrive/ESWC-figures/fewrel/KMmeans_CRE_fewrel_{0}/task_{1}_seen_task.json".format(experiment_id, task_id)
      out_acc_path = "drive/MyDrive/ESWC-figures/fewrel/KMmeans_CRE_fewrel_{0}/task_{1}_seen_task_result.json".format(experiment_id, task_id)
    responses = []
    relations = []
    for j, item in enumerate(data):
      prompt = item['prompt']
      relations.append(item['relation'])
      prompt = f"[INST] {prompt} [\INST] ### Answer:"
      response = get_prediction(model, tokenizer, prompt)
      print('test:', j)
      if len(response) == 0:
          print("No response")
          responses.append("")
      else:
          response = response[0]
          responses.append({"predict":response})

    write_json(responses, out_pred_path)


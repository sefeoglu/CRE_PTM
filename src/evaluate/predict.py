import sys
import os
from utils import write_json, read_json
from sklearn.metrics import accuracy_score
import json

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_prediction(model,tokenizer, prompt, length=250):
    """_summary_

    Args:
        model : trained model on task
        tokenizer (): tokenizer model
        prompt (str): test prompt
        length (int, optional): length of the prediction. Defaults to 250.

    Returns:
        response: predicted result (output)
    """

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096,return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs, max_new_tokens=length)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return response

def evaluate_model(input_folder, out_folder, experiment_id, task_id, model, tokenizer, current_task=True):
    """_summary_

    Args:
        input_folder (str): directory to test dataset
        out_folder (str): directory to prediction folder to save them
        experiment_id (int): run id
        task_id (int): task index id
        model: trained model on the task 
        tokenizer: tokenizer 
        current_task (bool, optional): decide the evaluation will be on current task or all seen tasks. Defaults to True.
    """
    if current_task:
      # current task accuracy
      input_path = input_folder+"/run_{0}/{1}/test_1.json".format(experiment_id, task_id)
      data = read_json(input_path)
      out_pred_path = out_folder+"/run_{0}/task_{1}_current_task_pred.json".format(experiment_id, task_id)
      out_acc_path = out_folder+ "/run_{0}/task_{1}_current_task_result.json".format(experiment_id, task_id)
    else:
     
      data = []
      for i in range(1, task_id+1):
        input_path = input_folder+"/run_{0}/task{1}/test_1.json".format(experiment_id, i)
        data.extend(read_json(input_path))
      out_pred_path = out_folder+"/run_{0}/task_{1}_seen_task.json".format(experiment_id, task_id)
      out_acc_path = out_folder+"/run_{0}/task_{1}_seen_task_result.json".format(experiment_id, task_id)
    responses, relations = [], []

    for j, item in enumerate(data):
      
      prompt = item['prompt']
      relations.append(item['relation'])
      response = get_prediction(model, tokenizer, prompt)
    #   print('test:', j)
      if len(response) == 0:
          print("No response")
          responses.append("")
      else:
        #   print(response[0])
          responses.append({"predict":response[0]})

    y_true = relations
    preds = [line['predict'] for line in responses]

    acc = accuracy_score(y_true, preds)
    result = [{"acc":acc}]

    write_json(responses, out_pred_path)
    write_json(result,out_acc_path)
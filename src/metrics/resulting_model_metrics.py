import json
from sklearn.metrics import accuracy_score

import numpy as np
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_model(experiment_id):

    results = []
    all_acc=0
    for experiment_id in range(1, 6):
      pred_path = f"super_clean_mistral_fewrel.json"
      preds = read_json(pred_path)
      preds = [line['relation'] for line in preds]
      start_index=0
      end_index=0
      total_acc = 0
      for i in range(1, 11):

        input_path = "llama_format_data/test/run_{0}/task{1}/test_1.json".format(experiment_id, i)
        print(input_path)
        y_true = read_json(input_path)
        y_trues = [line['relation'] for line in y_true]
        end_index += len(y_trues)
        print("start:{0}".format(start_index))
        print("end:{0}".format(end_index))
        task_preds = preds[start_index:end_index]

        print(len(task_preds))
        print(len(y_trues))
        acc = accuracy_score(y_trues, task_preds)

        row ={"acc":acc, "task_id":i, "experiment_id":experiment_id, "size":len(y_trues)}
        results.append(row)
        start_index = end_index
        total_acc += acc
      print("average_acc: {0}".format(total_acc/10))
      all_acc += total_acc/10
    print("all_acc: {0}".format(all_acc/1))
    results_metrics = {"results":results, "all_acc":all_acc/1}
    return results_metrics

def whole_acc(results):

    tot_whole_acc=0
    for i in range(1, 5):
    # Iterate through the list of results in results["results"]
        run_results = results["results"]
        run_results = [result for result in run_results if result['experiment_id'] == i]
        cum = 0
        size_tot = 0
        for item in run_results:
            cum+= item['acc']*item['size']
            size_tot += item['size']

        whole_acc = cum/size_tot
        tot_whole_acc +=whole_acc
        print(whole_acc)
    model_whole_acc = tot_whole_acc/5
    return model_whole_acc
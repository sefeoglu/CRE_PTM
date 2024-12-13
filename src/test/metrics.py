from sklearn.metrics import accuracy_score
import sys
import os
import json
def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write a json file to the given path."""
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
results = []

for run_id in range(1, 6):
    for task_id in range(1, 11):
        
        pred_file = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_fewrel/zero-shot/result_task_{0}_{1}.json".format(run_id, task_id)
        preds = read_json(pred_file)
        y_true = [line['ground_truth'] for line in preds]
        pred = [line['predict'] for line in preds]
        acc = accuracy_score(y_true, pred)
        print(acc)
        result = {"run":run_id, "task":task_id,  "acc":acc}
        results.append(result)

out_path = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_fewrel/zero_shot_fewrel_t5base_metrics.json"
write_json(out_path, results)
        

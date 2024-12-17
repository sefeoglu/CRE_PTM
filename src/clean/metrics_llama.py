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
gt_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/llama_format_data/test/"

for run_id in range(1, 6):
    for task_id in range(2, 11):
        y_trues = []
        for i in range(1, task_id+1):
            gt_file = gt_path+"run_{0}/task{1}/test_1.json".format(run_id, i)
            y_true =[ item['relation'] for item in read_json(gt_file)]
            y_trues.extend(y_true)

        pred_file = "/Users/sefika/phd_projects/CRE_PTM/src/clean/llama_results_clean/m_10/KMmeans_CRE_tacred_{0}/task_{1}_seen_task.json".format(run_id, task_id)
        preds = read_json(pred_file)
        
        # print(y_true)
        print(pred_file)
        pred = [str(line['clean']) for line in preds]
        
        acc = accuracy_score(y_trues, pred)
        print(acc)
        result = {"run":run_id, "task":task_id,  "acc":acc}
        results.append(result)

out_path = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_tacred/llama_metric_seen_m10.json"
write_json(out_path, results)
        

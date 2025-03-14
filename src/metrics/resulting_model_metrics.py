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


def evaluate_model( folder_path):

    results = []
    all_acc=0
    for experiment_id in [1,2,3,5]:
      pred_path = f"{folder_path}/KMmeans_CRE_tacred_{experiment_id}_extracted/task_10_seen_task.json"
      preds = read_json(pred_path)
      print(len(preds))
      preds = [line['predict'] for line in preds]
      start_index=0
      end_index=0
      total_acc = 0
      for i in range(1, 11):

        input_path = f"/Users/sefika/phd_projects/CRE_PTM/data/fewrel/llama_format_data/test/run_{experiment_id}/task{i}/test_1.json"
        # print(input_path)
        y_true = read_json(input_path)
        y_trues = [line['relation'] for line in y_true]
        end_index += len(y_trues)
        # print("start:{0}".format(start_index))
        # print("end:{0}".format(end_index))
        task_preds = preds[start_index:end_index]

        # print(len(task_preds))
        # print(len(y_trues))
        filtered_data = [(p, g) for p, g in zip(task_preds, y_trues) if p is not None and g is not None]
        # If filtered_data is empty, set pred_relations_task_1_filtered and gt_relations_filtered to empty lists to avoid errors
        if not filtered_data:
            pred_relations_task_1_filtered = []
            gt_relations_filtered = []
        else:
            # Unzip the filtered data back into separate lists
            pred_relations_task_1_filtered, gt_relations_filtered = zip(*filtered_data)
        
        acc = accuracy_score(gt_relations_filtered, pred_relations_task_1_filtered)

        row ={"acc":acc, "task_id":i, "experiment_id":experiment_id, "size":len(y_trues)}
        
        results.append(row)
        start_index = end_index
        total_acc += acc
    #   print("average_acc: {0}".format(total_acc/10))
      all_acc += total_acc/10
    # print("mean avg_acc: {0}".format(all_acc/5))
    results_metrics = {"results":results, "mean_avg_acc":all_acc/4}
    return results_metrics

def whole_acc(results):

    tot_whole_acc=0
    # print(results)
    # return 0
    for i in [1,2,3,5]:
    # Iterate through the list of results in results["results"]
        run_results = results["results"]
        run_results = [result for result in run_results if result['experiment_id'] == i]
        cum = 0
        size_tot = 0
        for item in run_results:
            cum+= item['acc']*item['size']
            size_tot += item['size']

        whole_acc = cum/size_tot
        tot_whole_acc += whole_acc
        # print(whole_acc)
    model_whole_acc = tot_whole_acc/4
    model_whole_acc_dict = {"mean model_whole_acc":model_whole_acc}

    return model_whole_acc_dict

if __name__ == "__main__":
    ### TODO: Change the path#
    # Change the path#
    result_folder = "/Users/sefika/phd_projects/CRE_PTM/resulting_metrics/results/fewrel/llama_seen_clean_mist_code"
    acc_and_results = evaluate_model(result_folder)
    model_whole_acc = whole_acc(acc_and_results)
    write_json(acc_and_results, f"llama_fewrel_m_10_acc.json")
    
    write_json(model_whole_acc, f"llama_fewrel_whole_acc.json")
    print(acc_and_results['mean_avg_acc'])
    print(model_whole_acc)  

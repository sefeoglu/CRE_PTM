import json
from sklearn.metrics import accuracy_score
import configparser
import numpy as np
import os
import configparser

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

print(PREFIX_PATH)
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_model(folder_path, test_folder):
    """
    Evaluate the model on the test data and return the results.
    Args:
    - folder_path: The path to the folder containing the model predictions.
    - test_folder: The path to the folder containing the test data.
    Returns:
    - A dictionary containing the results and the mean average accuracy.
    """
    results = []
    all_acc=0
    for experiment_id in range(1, 6):
      pred_path = f"{folder_path}/model{experiment_id}/task_10_seen_task.json"
      preds = read_json(pred_path)
      print(len(preds))
      preds = [line['predict'] for line in preds]
      start_index=0
      end_index=0
      total_acc = 0
      for i in range(1, 11):

        input_path = f"{test_folder}/test/run_{experiment_id}/task{i}/test_1.json"
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

      all_acc += total_acc/10
    results_metrics = {"results":results, "mean_avg_acc":all_acc/5}
    return results_metrics

def whole_acc(results):
    """
    Calculate the whole accuracy of the model.
    Args:
    - results: The results of the model evaluation.
    Returns:
    - A dictionary containing the mean whole accuracy of the model.
    """

    tot_whole_acc=0
    # print(results)
    # return 0
    for i in range(1, 6):
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
    
    model_whole_acc = tot_whole_acc/5

    model_whole_acc_dict = {"mean model_whole_acc":model_whole_acc}

    return model_whole_acc_dict

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read( PREFIX_PATH + "config.ini")
    test_folder = config["METRICS"]["test_data_folder"]
    results_folder = config["METRICS"]["results_folder"]
    w_results_path = config["METRICS"]["w_result_file_path"]
    a_results_path = config["METRICS"]["a_result_file_path"]

    acc_and_results = evaluate_model(result_folder, test_folder)
    model_whole_acc = whole_acc(acc_and_results)
    write_json(acc_and_results, a_results_path)
    
    write_json(model_whole_acc, w_results_path)

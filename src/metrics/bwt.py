import json
import numpy as np
from sklearn.metrics import accuracy_score
import configparser
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

print(PREFIX_PATH)

def compute_accuracy(y_true, results):

    y_pred = [ line['predict'] for line in results]
    filtered_data = [(p, g) for p, g in zip(y_pred, y_true) if p is not None and g is not None]

    if not filtered_data:
        pred_relations_task_1_filtered = []
        gt_relations_filtered = []
    else:
        # Unzip the filtered data back into separate lists
        pred_relations_task_1_filtered, gt_relations_filtered = zip(*filtered_data)
    
    return accuracy_score(gt_relations_filtered, pred_relations_task_1_filtered)
    
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_results(result_folder_folder, test_data):
    """ Prepare Acc Matrix for BWT calculation
    Args:
    - input_folder: The path to the folder containing the model predictions.
    - test_data: The path to the folder containing the test data.
    Returns:
    - A 2D list containing the accuracies for each task in each run.
    """
    run = []
    for run_id in range(1, 6):
        results = []
        last_task_results = []

        for task_id in range(1, 10):
            task = result_folder_folder + f"/model{run_id}/task_{task_id}_current_task_pred.json"
            result = read_json(task)
            if len(result) == 1:
                results.append(result[0]['acc'])
            else:
                y_true_path = f"{test_data}/run_{run_id}/task{task_id}/test_1.json"
                y_true = [ item['relation'] for item in read_json(y_true_path)]
                acc = compute_accuracy(y_true, result)
                results.append(acc)

        task = result_folder_folder+f"/model{run_id}/task_10_seen_task.json"

        last_task = read_json(task)
        end = 0
        start = 0

        for task_id in range(1, 11):
            y_true_path = f"{test_data}/run_{run_id}/task{task_id}/test_1.json"
            y_true = read_json(y_true_path)
            y_true  = [line['relation'] for line in y_true]
            end = start + len(y_true)
            acc = compute_accuracy(y_true, last_task[start:end])
            start = end
            last_task_results.append(acc)

        results.append(last_task_results)
        
        run.append({'run_id':run_id, 'results':results})

    return run


def calculate_bwt(accuracies):
    """
    Calculate the Backward Transfer (BWT) metric.

    Parameters:
    - accuracies: A 2D list or numpy array where accuracies[N-1][t] is the test accuracy
                  on task t after sequential training on all N tasks, and accuracies[t][t]
                  is the test accuracy on task t immediately after it was learned.

    Returns:
    - BWT: The Backward Transfer metric.
    """
    # Convert to numpy array for easier indexing
    # print(accuracies)
    accuracies = np.array(accuracies)
    
    num_runs = accuracies.shape[0]
    # print(accuracies)
    # print(num_runs)
    # # Compute the backward transfer
    # # Changed the range to iterate up to the minimum of N-1 and num_runs
    for i in range(1, num_runs):
        accs = accuracies[i]['results'][:9]
        seen_task = accuracies[i]['results'][9]
        N = len(accs)
        bwt = 0
        for t in range(0, N):
            A_t_t = accs[t]  # Accuracy on task t after all tasks
            A_N_t = seen_task[t]
            bwt += (A_N_t - A_t_t)
        
        bwt /= (N - 1)  # Averaging the differences

    mean_bwt = bwt / num_runs
    # print(mean_bwt)

    return mean_bwt

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(PREFIX_PATH + 'config.ini')
    results_folder = config["METRICS"]["results_folder"]
    test_data = config["METRICS"]["test_data_folder"]
    bwt_file_path = config["METRICS"]["bwt_file_path"]
    accuracy_matrix = get_results(input_folder)
    # print(np.array(accuracy_matrix).shape)
    bwt = calculate_bwt(accuracy_matrix)
    write_json(bwt, bwt_file_path)
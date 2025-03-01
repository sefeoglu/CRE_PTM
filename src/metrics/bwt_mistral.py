import json
import numpy as np
from sklearn.metrics import accuracy_score
def compute_accuracy(y_true, results):
    y_pred = [ line['relation'] for line in results]
    return accuracy_score(y_true, y_pred)
    
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_results(input_folder):
    run = []
    for run_id in range(1, 6):
        results = []
        for task_id in range(1, 10):
            task = input_folder + f"/KMmeans_CRE_tacred_{run_id}_extracted/task_task{task_id}_current_task_pred.json"
            result = read_json(task)
            if len(result) == 1:
                results.append(result[0]['acc'])
            else:
                y_true_path = f"/Users/sefika/phd_projects/CRE_PTM copy/data/tacred/data/llama_format_data/test/run_{run_id}/task{task_id}/test_1.json"
                y_true = [ item['relation'] for item in read_json(y_true_path)]
                acc = compute_accuracy(y_true, result)
                results.append(acc)
        task_acc = input_folder+f"/KMmeans_CRE_tacred_{run_id}_extracted/task_10_seen_task.json"
        task_acc = read_json(task_acc)
        if len(task_acc) == 1:
            results.append(task_acc[0]['acc'])
        else:
            gt = []
            for task_id in range(1, 11):
                y_true_path = f"/Users/sefika/phd_projects/CRE_PTM copy/data/tacred/data/llama_format_data/test/run_{run_id}/task{task_id}/test_1.json"
                y_true = read_json(y_true_path)
                # print(y_true)
                gt.extend([line['relation'] for line in y_true])
            acc = compute_accuracy(gt, task_acc)
            results.append(acc)
        run.append(results)

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
    accuracies = np.array(accuracies)

    # Number of tasks
    print(accuracies)
    N = accuracies.shape[0]
    print(N)
    # Number of runs -- getting this from the shape of the accuracy matrix
    num_runs = accuracies.shape[1]
    print(num_runs)


    # Compute the backward transfer
    # Changed the range to iterate up to the minimum of N-1 and num_runs
    bwt = (1 / (N - 1)) * sum(accuracies[-1, t] - accuracies[t, t] for t in range(min(N - 1, num_runs)))

    return bwt

if __name__ == "__main__":
   ## TODO ##
   ## remove folder path ###
   input_folder = "/Users/sefika/phd_projects/CRE_PTM copy/src/test/results_memory_cl_tacred/llama_results/m_10"
   m5_accuracies = get_results(input_folder)
   print(np.array(m5_accuracies).shape)
   m5_bwt = calculate_bwt(m5_accuracies)
   print(m5_bwt)
#    input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m10"
#    m10_accuracies = get_results(input_folder)
#    m10_bwt = calculate_bwt(m10_accuracies)

#    input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m15"
#    m15_accuracies = get_results(input_folder)
#    m15_bwt = calculate_bwt(m15_accuracies)
   
#    bwt = [{"model":"t5", "m5":m5_bwt, "m10":m10_bwt,"m15":m15_bwt}]

   write_json(m5_bwt, "bwt_llama_tacred.json")
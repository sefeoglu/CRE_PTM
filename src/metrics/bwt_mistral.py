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
        last_task_results = []

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

        task = input_folder+f"/KMmeans_CRE_tacred_{run_id}_extracted/task_10_seen_task.json"

        last_task = read_json(task)
        end=0
        start=0

        for task_id in range(1, 11):
            y_true_path = f"/Users/sefika/phd_projects/CRE_PTM copy/data/tacred/data/llama_format_data/test/run_{run_id}/task{task_id}/test_1.json"
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
    print(accuracies)
    print(num_runs)


    # # Compute the backward transfer
    # # Changed the range to iterate up to the minimum of N-1 and num_runs
    for i in range(1, num_runs):
        accs = accuracies[i]['results'][:9]
        seen_task = accuracies[i]['results'][9]
        N = len(accs)
        bwt = 0
        for t in range(0, N):
            A_N_t = accs[t]  # Accuracy on task t after all tasks
            A_t_t = seen_task[t]
            bwt += (A_N_t - A_t_t)
        
        bwt /= (N - 1)  # Averaging the differences

    mean_bwt = bwt / num_runs
    print(mean_bwt)

    return mean_bwt

if __name__ == "__main__":
   ## TODO ##
   ## remove folder path ###
   input_folder = "/Users/sefika/phd_projects/CRE_PTM copy/src/test/results_memory_cl_tacred/mistal_results/m_15"
   m5_accuracies = get_results(input_folder)
   print(np.array(m5_accuracies).shape)
   m5_bwt = calculate_bwt(m5_accuracies)
#    print(m5_bwt)
#    input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m10"
#    m10_accuracies = get_results(input_folder)
#    m10_bwt = calculate_bwt(m10_accuracies)

#    input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m15"
#    m15_accuracies = get_results(input_folder)
#    m15_bwt = calculate_bwt(m15_accuracies)
   
#    bwt = [{"model":"t5", "m5":m5_bwt, "m10":m10_bwt,"m15":m15_bwt}]

   write_json(m5_bwt, "bwt_mistral_tacred_15.json")
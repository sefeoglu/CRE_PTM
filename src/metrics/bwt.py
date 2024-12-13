import json
import numpy as np
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_results(input_folder):
    run = []
    for task_id in range(1, 10):
        results = []
        for run_id in range(1, 6):
            task = input_folder+"/run_{0}/task_task{1}_current_task_result.json".format(run_id, task_id)
            result = read_json(task)
            results.append(result[0]['acc'])
        run.append(results)
    task_10 =[]

    for run_id in range(1, 6):
        task = input_folder+"/run_{0}/task_{1}_seen_task_result.json".format(run_id, task_id)
        last = read_json(task)
        task_10.append(last[0]['acc'])

    run.append(task_10)
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
   
   input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m5"
   m5_accuracies = get_results(input_folder)
   print(np.array(m5_accuracies).shape)
   m5_bwt = calculate_bwt(m5_accuracies)

   input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m10"
   m10_accuracies = get_results(input_folder)
   m10_bwt = calculate_bwt(m10_accuracies)

   input_folder = "/Users/sefika/phd_projects/CRE_PTM/results/cre_all_results/results_memory_cl_tacred/flan_t5/m15"
   m15_accuracies = get_results(input_folder)
   m15_bwt = calculate_bwt(m15_accuracies)
   
   bwt = [{"model":"t5", "m5":m5_bwt, "m10":m10_bwt,"m15":m15_bwt}]

   write_json(bwt, "bwt_flan_t5.json")
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
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



# Function to extract relations
def find_relations(data, relations):
        relations = [ relation for relation in relations]
        for relation in relations:
            if relation in data:
                return relation

def clean(data, task_relations):
    delimiter = "### Answer:"
    answers = []
    for i, item in enumerate(data):
        answer = item['original'].split(delimiter)[1]
        answer = answer.split("\n\n")[0].replace("#","").rstrip()
        
        # answer = find_relations(answer, task_relations)
        answer = ' '.join(answer.split())

        line = {"id":i, "predict":answer, "original":item['original']}
        answers.append(line)

    return answers

if __name__ == "__main__":

    input_folder_path = "/Users/sefika/phd_projects/CRE_PTM/src/clean/llama_results_clean/m_10"
    out_folder_path = "/Users/sefika/phd_projects/CRE_PTM/src/clean/llama_results_clean/m_10_clean/"
    tasks_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/related_work_results/resluts/tacred_tasks.json"
    tasks = read_json(tasks_path)

    for run_id in range(1, 6):
        run = "run_{0}".format(run_id)
        run_tasks = tasks[run]
        seen_relations = []
        for i, item in enumerate(run_tasks):
            seen_relations.extend(run_tasks[item])
        for task_id in range(2, 11):
            task_del = "task{0}".format(task_id)
            task_relations = run_tasks[task_del]
            # seen_relations.extend(task_relations)
            input_path = input_folder_path+f"/KMmeans_CRE_tacred_{run_id}/task_{task_id}_seen_task.json"
            out_path = out_folder_path+f"/m_10_clean/KMmeans_CRE_tacred_{run_id}/task_{task_id}_seen_task.json"
            data = read_json(input_path)
            cleaned_data = clean(data,task_relations)
            write_json(out_path, cleaned_data)





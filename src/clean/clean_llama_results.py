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
    delimiter = "Answer:"
    ## Answer: followed_by ###
    answers = []
    for i, item in enumerate(data):
        answer = item['predict'].split(delimiter)[1].split("#")[0]
        answer = answer.replace("\n", " ")
        answer = answer.replace("###", "").rstrip()
        # answer = answer.replace("Answer:", "")
        # print(task_relations)
        # answer = find_relations(answer, task_relations)
        try:
            print("Answer:", answer)
        except:
            answer = ""

        line = {"id":i, "predict":answer, "original":item['predict']}
        answers.append(line)

    return answers



if __name__ == "__main__":
    ## TODO ##
    ## : Change the paths ##
    input_folder_path = "/Users/sefika/phd_projects/CRE_PTM/resulting_metrics/results/fewrel/llama_seen/"
    out_folder_base_path = "/Users/sefika/phd_projects/CRE_PTM/resulting_metrics/results/fewrel/llama_seen/llama_cleaned/"
    tasks_path = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/fewrel10tasks.json"
    tasks = read_json(tasks_path)
    relations_path = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/pid2name.json"
    relations = read_json(relations_path)

    for run_id in range(1, 6):
        run = "run_{0}".format(run_id)
        run_tasks = tasks[run]
        
        
        for task_id in range(1, 11):
            seen_relations = []
            task_del = "task_{0}".format(task_id)
            task_relations = run_tasks[task_del]
            task_name = [relations[relation][0].replace(" ", "_") for relation in task_relations]
            seen_relations.extend(task_name)

            input_path = input_folder_path  + f"KMmeans_CRE_tacred_{run_id}/task_{task_id}_current_task_pred.json"
            data = read_json(input_path)

            out_folder_path = out_folder_base_path + f"KMmeans_CRE_tacred_{run_id}/"
            out_path = out_folder_path +  f"task_{task_id}_current_task_pred.json"

            
            cleaned_data = clean(data,seen_relations)
            write_json(out_path, cleaned_data)





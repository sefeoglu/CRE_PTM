from sklearn.metrics import accuracy_score
import sys
import os
import json
import configparser

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

print(PREFIX_PATH)

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

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH + 'config.ini')
    results_path = config['TEST']['results_path']
    out_folder_path = config['TEST']['clean_results_folder']
    tasks_path = config['TEST']['tasks_path']
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
            input_path = input_folder_path+f"/model{run_id}/task_{task_id}_seen_task.json"
            out_path = out_folder_path+f"/model{run_id}/task_{task_id}_seen_task.json"
            data = read_json(input_path)
            cleaned_data = clean(data,task_relations)
            write_json(out_path, cleaned_data)





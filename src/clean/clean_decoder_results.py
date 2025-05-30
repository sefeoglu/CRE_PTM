import os
import re
import json

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
# Sample data


def extact_answers(data, relation_types):
    for relation in relation_types:
        if relation in data:
            return relation
    return ''

def clean(data, task_relations_path, model_name='llama'):
    task_relations = read_json(task_relations_path)
    relation_types = [item['relation'].replace(' ', "_") for item in task_relations]
    relation_types = list(set(relation_types))
    
    delimiter = "Answer:"
    answers = []
    for i, item in enumerate(data):
        item['predict'] = item['predict'].replace("Answer: \n",delimiter)
        item['predict'] = item['predict'].replace("Answer:\n",'Answer:')
        answer = item['predict'].split(delimiter)[1]
        if model_name != 'llama':
            answer = answer.split("\n")[0].replace("#","").rstrip()
        answer = ' '.join(answer.split())
        answer = extact_answers(answer, relation_types)
        line = {"id":i, "predict":answer, "original":item['predict']}
        answers.append(line)

    return answers

def get_answers(folder_path, task_relations_path):
 
    try:
        for i in range(1, 6):

            folder = f'{folder_path}_{i}'
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                print(file)
                if file != ".ipynb_checkpoints" :
                    data =  read_json(file_path)
                    extracted_answers = clean(data, task_relations_path)
                    out_path = os.path.join(folder+'_extracted', file)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    write_json(extracted_answers, out_path)

    except Exception as e:
        print(f"file name: {file} ---> run {i}")
        print(e)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(PREFIX_PATH + 'config.ini')
    results_path = config['TEST']['results_path']
    tasks_path = config['TEST']['tasks_path']
    get_answers(results_path, tasks_path, model_name='llama')

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

def main(input_file, out_file, task_relation_file):
    task_relations = read_json(task_relation_file)
    input_data = read_json(input_file)
    
    task_data = []
    for _, line in enumerate(input_data):
       
        if line['relation'] in task_relations:
            task_data.append(line)
            print(line['relation'])

    write_json(out_file, task_data)

if __name__ == "__main__":
    for i in range(1, 11):
        input_file = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/dev.json"
        output_file = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/train_tasks/task{0}/dev.json".format(i)
        task_relation_file = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/test_sets/relations/task{0}.json".format(i)
        main(input_file, output_file, task_relation_file)

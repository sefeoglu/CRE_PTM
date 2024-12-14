import sys
import os
import json
import random
from sklearn.model_selection import train_test_split

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


def get_prompt(sentence, head, tail, relations, prompt_type):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    relations = ", ".join([relation.replace(" ","_") for relation in relations])

    if not prompt_type:
        template2_zero_shot = """Sentence: """ + str(sentence)+ """\n""" +\
                            """ What is the relation type between """+head+""" and """+tail+""" according to given relationships below in the following sentence?\n""" +\
                            """ Relation types: """ + relations + """. \n""" +\
                            """ Answer:"""
    else:
        template2_zero_shot = """Sentence: """ + str(sentence)+ """\n""" +\
                            """ What is the relation type between """+head+""" and """+tail+"""  according to given relationships below in the following sentence?\n""" +\
                            """ Relation types: """ + relations + """. \n"""

    return template2_zero_shot

def main(task_train_data, relations, task_relations, task_id, run_id, out_folder,relation_id, prompt_type=False):

    data = {"train":task_train_data}
    
    for key, value in data.items():
    
        train_out_file_path = out_folder+"train/run_{0}/task{1}/train_1.json".format(run_id, task_id)
        test_out_file_path = out_folder+"test/run_{0}/task{1}/test_1.json".format(run_id, task_id)
        val_out_file_path = out_folder+"train/run_{0}/task{1}/dev_1.json".format(run_id, task_id)
        prompts = []
        train_selected_data = []
        test_data_selected = []
        val_data_selected = []
        for relation in task_relations:
            
            relation_data = [line for line in value if relation == line['relation_PID']]
            if len(relation_data)>0:
                # print(len(relation_data))
                train_data, test_data = train_test_split(relation_data, test_size=140, random_state=42)
                train_data, val_data = train_test_split(train_data, test_size=140, random_state=42)
                train_selected_data.extend(train_data)
                val_data_selected.extend(val_data)
                test_data_selected.extend(test_data)
                print("len: {0}".format(len(test_data)))
        prompts = []
        for line in train_selected_data:
            relations_list = [relation_id[PID][0] for PID in relations]
            input = {"prompt":get_prompt(line['sentence'], line['subject'], line['object'], relations_list, prompt_type), "relation": line['relation'].replace(" ","_")}
            prompts.append(input)
            # print(len(relations))
        write_json(train_out_file_path, prompts)
        prompts = []
        for line in val_data_selected:
            relations_list = [relation_id[PID][0] for PID in relations]
            input = {"prompt":get_prompt(line['sentence'], line['subject'], line['object'], relations_list, prompt_type), "relation": line['relation'].replace(" ","_")}
            prompts.append(input)
            # print(len(relations))
        write_json(val_out_file_path, prompts)

        for i in range(task_id+1, 11):
            print(task_id)
            out_file_path = out_folder+"train/run_{0}/task_memory_{1}/dev_{2}.json".format(run_id, i, task_id)
            print(out_file_path)
            write_json(out_file_path, prompts)
        
        prompts = []
        for line in test_data_selected:
            relations_list = [relation_id[PID][0] for PID in relations]
            input = {"prompt":get_prompt(line['sentence'], line['subject'], line['object'], relations_list, prompt_type), "relation": line['relation'].replace(" ","_")}
            prompts.append(input)

            # print(len(relations))
        write_json(test_out_file_path, prompts)

  


if __name__ == "__main__":
    all_train_data  = read_json("/Users/sefika/phd_projects/CRE_PTM/data/fewrel/final/train_wiki.json")
    all_tasks = read_json("/Users/sefika/phd_projects/CRE_PTM/data/fewrel/fewrel10tasks.json")
    out_folder = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/llama_format_data/"
    relation_id = read_json("/Users/sefika/phd_projects/CRE_PTM/data/fewrel/data/pid2name.json")
    relations=[]
    for run_id in range(1,6):

        run_name = "run_{0}".format(run_id)
        run_tasks = all_tasks[run_name]

        for task_id in range(1, 11):
            task_name = "task_{0}".format(task_id)
            task_relations = run_tasks[task_name]
            relations.extend(task_relations)
            
            file = out_folder+"relations/run_{0}/task{1}.json".format(run_id, task_id)
            relations_list = [relation_id[PID][0].replace(" ", "_") for PID in relations]
            write_json(file, relations_list)
            task_train_data = [item for item in all_train_data if item['relation_PID'] in task_relations]
    
         
            main(task_train_data, task_relations, task_relations,task_id, run_id, out_folder, relation_id, True)
    # print(len(relations))
import sys
import os
import json
import random
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
def get_prompt(sentence, head, tail, relations,  prompt_type):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    relations = ", ".join([relation for relation in relations])

    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Question: What is the relation type between tail and head entitiesaccording to given relationships below in the following sentence?\n""" +\
                        """ Query Sentence: """ + str(sentence)+ """\n""" +\
                        """ head: """ + head + """. \n""" +\
                        """ tail: """ + tail + """. \n""" +\
                        """ Relation types: """ + relations + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot

def main(task_train_data, task_dev_data, task_test_data, old_dev, old_test, relations, task_relations, task_id, run_id, out_folder,  prompt_type=False):

    data = {"train":task_train_data, "dev":task_dev_data}
    selected_data = []
    for key, value in data.items():
        out_file_path = out_folder+"train/run_{0}/task{1}/{2}_1.json".format(run_id, task_id, key)
        prompts = []
        selected_data = []
        for relation in task_relations:
            relation_data = [line for line in value if relation == line['relation']]
            selected_data.extend(relation_data)

        if key == "dev":
            # selected_data.extend(old_dev)
            task_dev_data = selected_data
        else:
            task_train_data = selected_data

        relation_type = []
        # print(selected_data[0])
        # break
        for line in selected_data:
            prompt = get_prompt(line['sentence'], line['subject'], line['object'], relations, prompt_type)
            relation_type.append(line['relation'])
            prompts.append({'id':line['id'], 'prompt':prompt, 'relation':line['relation']})
        
        print(len(relation_type))
        write_json(out_file_path, prompts)

    prompts = []
    out_test_file_path = out_folder+"test/run_{0}/task{1}/test_1.json".format(run_id, task_id)
    selected_test_data = []

    for relation in task_relations:
        test_relation_data = [line for line in task_test_data if line['relation']==relation]
        if len(test_relation_data)>40:
            ids = [line['id'] for line in test_relation_data]
            selected_ids = random.sample(ids, 40)
            
            test_relation_data = [ line for line in task_test_data if line["id"] in selected_ids]
            selected_test_data.extend(test_relation_data)
        else:
            selected_data.extend(test_relation_data)
    task_test_data = selected_test_data
    # selected_test_data.extend(old_test)
        
    for line in selected_test_data:
        prompt = get_prompt(line['sentence'], line['subject'], line['object'], relations,  prompt_type)
        prompts.append({'id':line['id'], 'prompt':prompt, 'relation':line['relation']})

    write_json(out_test_file_path, prompts)
    return task_dev_data, task_test_data

if __name__ == "__main__":
    all_train_data  = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/train.json")
    all_test_data = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/test.json")
    all_dev_data = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/dev.json")
    all_tasks = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/related_work_results/resluts/tacred_tasks.json")
    out_folder = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/llama_prompt_template/"
    
    for run_id in range(1,6):

        run_name = "run_{0}".format(run_id)
        run_tasks = all_tasks[run_name]
        incremental_tasks = []
        old_dev, old_test = [],[]
        for task_id in range(1, 11):
            task_name = "task{0}".format(task_id)
            task_relations = run_tasks[task_name]
            # incremental_tasks.extend(task_relations)
            file = out_folder+"relations/run_{0}/task{1}.json".format(run_id, task_id)
            # write_json(file, incremental_tasks)
            # print
            task_train_data = [item for item in all_train_data if item['relation'] in task_relations]
            task_test_data = [item for item in all_test_data if item['relation'] in task_relations]
            task_dev_data = [item for item in all_dev_data if item['relation'] in task_relations]
            old_dev, old_test = main(task_train_data, task_dev_data, task_test_data, old_dev, old_test, task_relations, task_relations,task_id, run_id, out_folder)
            
        # break
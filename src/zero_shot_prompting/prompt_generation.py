import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import ast
import configparser
import datetime
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_prompt_template(sentence, relation, head, tail, context):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """

    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Question : What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ head: """ + head + """. \n""" +\
                        """ tail: """ + tail + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot
def get_prompt_template_relatedwork(sentence, relations, head, tail, sample):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    template2_zero_shot =""" Example Sentence"""+str(sample['sentence'])+"""\n"""+\
                        """ The relation between """+sample['head']+""" and """+sample['tail']+""" is """+sample['relation']+"""\n"""+\
                        """ Sentence: """ + str(sentence)+ """\n""" +\
                        """ What is the relation type between """+head+""" and """+tail+""" entities according to given relationships below in the following sentence?\n""" +\
                        """ Relation types: """ + relations + """. \n""" +\
                        """ Answer:"""
    return template2_zero_shot
def compute_sentence_embeddings(data):
    """Compute the sentence embeddings for the sentences in the dataset
    Args:
        data (list): list of sentences
    Returns:
        list: list of sentence embeddings
    """
    sent_embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("The embeddings will be completed for {0} sentences".format(len(data)))
    
    for i, line in enumerate(data):
        if not "token" in line.keys():
            sent = line['sentence']
        else:
            sent = ", ".join(line['token'])
        embeddings = model.encode(sent)
        sent_embeddings.append(embeddings)
        # print("Processed sentence: ", i)

    print("The embeddings were completed for {0} sentences".format(len(sent_embeddings)))

    return sent_embeddings


def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    similarities = []

    for test_index, test_item in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            print(test_item)
            if train_line['relation'] == test_item['relation']:
                sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
                # print(train_line)
                train_sentence = " ".join(train_line['token'])
                    
                context =  train_sentence
                train_similarities.append({"train":train_index, "simscore": sim, "sentence":context,"head":train_line['subject'], "tail":train_line['object'],"relation":train_line['relation']})
        
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
        # print(train_similarities)
        if len(train_similarities) > 0:
            row = {"test":test_index, 
                "sentence":train_similarities[0]['sentence'],
                "train_idex":train_similarities[0]['train'], 
                "simscore":float(train_similarities[0]['simscore']),
                'head':train_similarities[0]['head'],
                'tail':train_similarities[0]['tail'],
                'relation':train_similarities[0]['relation']
                }
            similarities.append(row)


        # print("test index: ", test_index)

    return similarities

def dynamic_task_prompt_generation(test_data, similarities, tasks):

    relation_types = ", ".join(tasks)

    prompt_list = []
    for sim in similarities:
        test_item_idx = sim['test']
        test_item = test_data[test_item_idx]
        sentence = test_item['sentence']
        relation = test_item['relation']
        head = test_item['subject']
        tail = test_item['object']
        context = sim
        # print(relation_types)
        prompt = get_prompt_template_relatedwork(sentence, relation_types, head, tail, context)
        row_data = {"prompt":prompt, "relation":relation}
        prompt_list.append(row_data)

    return prompt_list


def main(task_path, train_data, test_data_path, full_test_path, all_relations):
    print(datetime.datetime.now())
    selected_tasks = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/id2rel_tacred.json")
    all_relations = read_json("/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/train.json")
    all_relations = [item['relation'] for item in all_relations]
    all_relations = list(set(all_relations))
    full_test_data = read_json(full_test_path)
    
    base_relations = [item for item in all_relations if not item in selected_tasks]
    base_relations = list(set(base_relations))
    base_data = [ line for line in train_data if line['relation'] in base_relations ]
    # print(base_relations)
 
    # base_embdb_data = compute_sentence_embeddings(base_data)
    for run_id in range(1, 6):
        out_folder = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/dynamic_rag4re_extended_same_setting_one_shot_context/run_{0}/".format(run_id)
        run_number = "run_{0}".format(run_id)
        run_tasks = read_json(task_path)[run_number]
        incremental_relation_types = []
        for i in range(1,11):
            task_id = "task{0}".format(i)
            task = run_tasks[task_id]
            incremental_relation_types.extend(task)

            task_train_data = [item for item in train_data if item['relation'] in task]
            
            task_train_embDB = compute_sentence_embeddings(task_train_data)
            # task_train_embDB.extend(base_embdb_data)
            test_data_file = test_data_path+"run_{0}/task{1}/test_1.json".format(run_id, i)
            task_test_data_ids = [item['id'] for item in read_json(test_data_file)]
            
            task_test_data = [ item for item in full_test_data if item['id'] in task_test_data_ids]
            
            task_test_embs = compute_sentence_embeddings(task_test_data)
            similarities = compute_similarity(task_test_data, task_train_data, task_train_embDB,task_test_embs)
            print(datetime.datetime.now())

            task_prompts = dynamic_task_prompt_generation(task_test_data, similarities, incremental_relation_types)
            out_path = out_folder+"prompt_one_shot_task{0}.json".format(i)
            write_json(out_path, task_prompts)
        
        

        

if __name__ == "__main__":
    ### compute embeddings for tasks incrementally.
    ### compute similarities incrementally
    ### generate prompts for tasks
    train_data_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/train.json"
    train_data = read_json(train_data_path)
    task_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/related_work_results/resluts/tacred_tasks.json"
    test_data_path = '/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/memory_based_same_cre/test/'
    full_test_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/test.json"
    # out_file_base = ""
    all_relations = [ item['relation'] for item in train_data]

    main(task_path, train_data, test_data_path, full_test_path, all_relations)
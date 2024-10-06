import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
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
                        """ Example Sentence: """+ str(context)+ """\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ head: """ + head + """. \n""" +\
                        """ tail: """ + tail + """. \n""" +\
                        """ Relation types: """ + relation + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot


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
        sent = " ".join(line['token'])
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

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_sentence = " ".join(train_line['token'])
                
            context =  train_sentence
            train_similarities.append({"train":train_index, "simscore": sim, "sentence":context})

        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        # print("test index: ", test_index)

    return similarities

def dynamic_task_prompt_generation(test_data, similarities, task_file):

    prompt_list = []
    tasks = open(task_file, "r").readlines()
    relation_types = ", ".join(tasks)

    for sim in similarities:
        test_item_idx = sim['test']
        test_item = test_data[test_item_idx]
        sentence = test_item['sentence']
        relation = test_item['relation']
        head = test_item['subject']
        tail = test_item['object']
        context = sim['similar_sentence']
        prompt = get_prompt_template(sentence, relation_types, head, tail, context)
        row_data = {"prompt":prompt, "relation":relation}
        prompt_list.append(row_data)

    return prompt_list


def main(task_path, train_data, test_data_path):
    print(datetime.datetime.now())
    
    for i in range(1,11):
        task_file  = task_path+"task{0}.json".format(i)
        task = open(task_file, "r").readline()
        
        task_train_data = [item for item in train_data if item['relation'] in task]
        
        task_train_embDB = compute_sentence_embeddings(task_train_data)

        task_test_path = test_data_path+str(i)+"/test.json"
        task_test_data = read_json(task_test_path)
        task_test_embs = compute_sentence_embeddings(task_test_data)
        similarities = compute_similarity(task_test_data, task_train_data, task_train_embDB,task_test_embs)
        print(datetime.datetime.now())

        task_prompts = dynamic_task_prompt_generation(task_test_data, similarities, task_file)
        out_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/dynamic_rag4re/prompt_task{0}.json".format(i)
        write_json(out_path, task_prompts)
        # break

        

if __name__ == "__main__":
    ### compute embeddings for tasks incrementally.
    ### compute similarities incrementally
    ### generate prompts for tasks
    train_data_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/train.json"
    train_data = read_json(train_data_path)
    task_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/test_sets/relations/"
    test_data_path = '/Users/sefika/phd_projects/CRE_PTM/data/tacred/test_sets/task'
    # out_file_base = ""
    main(task_path, train_data, test_data_path)
import sys
import os
import json


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"
print(PREFIX_PATH)
import configparser
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

def prepare_data_re(data, relation_id_path):
    """ Prepare data for FewRel dataset."""
    """Args:
    data: A dictionary with key as relation and value as list of sentences.
    relation_id_path: A path to the json file containing relation id.
    Returns:
    dataset: A list of dictionaries containing sentence, subject, object, relation, relation_PID.
    """
    dataset = []
   
    relation_id = read_json(relation_id_path)
  
    for i, relation in enumerate(data.keys()):
        sentences = data[relation]
      
        for line in sentences:

            tokens = line['tokens']
            print(line['h'][2])
            print(tokens[line['h'][2][0][0]])

            sentence = " ".join([t for t in tokens])
            head = " ".join([ tokens[id] for id in line['h'][2][0]])
            tail = " ".join([tokens[id] for id in line['t'][2][0]])
            raw_data = {
                "sentence":sentence,
                "tokens":tokens,
                "subject":head,
                "object":tail,
                "relation": relation_id[relation][0],
                "relation_PID":relation
            }
            dataset.append(raw_data)

    return dataset

def main(file_path, out_path, relation_id):
    """ Main function to prepare FewRel dataset."""
    data =  read_json(file_path, relation_id)
    write_json(out_path, dataset)

    
if __name__ =="__main__":
    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+'config.ini')
    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    relation_id = config['DATAPREPARATION']['relation_id']
    main(file_path, out_path, relation_id)
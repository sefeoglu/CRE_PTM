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

def prepare_data_re(data):
    dataset = []

    for i, line in enumerate(data):
        relation = line['relation']
        token = line['token']
        sentence = " ".join([t for t in token])
        subject_entity = " ".join( token[int(line['subj_start']):int(line['subj_end'])+1])
        object_entity = " ".join(token[int(line['obj_start']):int(line['obj_end'])+1])
        subj_type = line['subj_type']
        obj_type = line['obj_type']
        raw_data = {
                    "id":line['id'],
                    "sentence": sentence,
                    "token": token,
                    "subject": subject_entity,
                    "subject_type": subj_type,
                    "object": object_entity,
                    "object_type": obj_type,
                    "relation":relation
                    }
        dataset.append(raw_data)
    return dataset

def main(file_path, out_path):
    data =  read_json(file_path)
    dataset = prepare_data_re(data)
    write_json(out_path, dataset)

    
if __name__ =="__main__":

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+'config.ini')
    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    
    main(input_file, output_file)
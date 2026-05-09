import configparser
from pathlib import Path

from src.utils import read_json, write_json

def prepare_data_re(data):
    dataset = []

    for line in data:
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
    write_json(dataset, out_path)

    
if __name__ =="__main__":

    config = configparser.ConfigParser()
    project_root = Path(__file__).resolve().parents[2]
    config.read(project_root / 'config.ini')
    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    
    main(input_file, output_file)

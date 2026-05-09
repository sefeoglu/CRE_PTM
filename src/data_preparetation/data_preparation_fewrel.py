import configparser
from pathlib import Path

from src.utils import read_json, write_json

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
  
    for relation in data.keys():
        sentences = data[relation]
       
        for line in sentences:

            tokens = line['tokens']
            sentence = " ".join([t for t in tokens])
            head_entity = " ".join([tokens[token_id] for token_id in line['h'][2][0]])
            tail_entity = " ".join([tokens[token_id] for token_id in line['t'][2][0]])
            raw_data = {
                "sentence":sentence,
                "tokens":tokens,
                "subject":head_entity,
                "object":tail_entity,
                "relation": relation_id[relation][0],
                "relation_PID":relation
            }
            dataset.append(raw_data)

    return dataset

def main(file_path, out_path, relation_id):
    """ Main function to prepare FewRel dataset."""
    data = read_json(file_path)
    dataset = prepare_data_re(data, relation_id)
    write_json(dataset, out_path)

    
if __name__ =="__main__":
    config = configparser.ConfigParser()
    project_root = Path(__file__).resolve().parents[2]
    config.read(project_root / 'config.ini')
    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    relation_id = config['DATAPREPARATION'].get('relation_id') or config['DATAPREPARATION'].get('relation_ids')
    main(input_file, output_file, relation_id)

import configparser
from pathlib import Path
from src.data_preparetation import (
    data_preparation_fewrel,
    data_preparation_tacred,
    instruction_ft_data_same_setting_fewrel,
    instruction_ft_data_same_setting_tacred,
)

from src.CRE import trainer_t5, trainer_decoder


def _get_dataset_name(config):
    return config.get("DATA", "dataset", fallback=config.get("DATASET", "dataset_name", fallback=""))


def _get_model_id(config):
    model_id = config.get("MODEL", "model_id", fallback="")
    if model_id:
        return model_id
    model_name = config.get("MODEL", "model_name", fallback="").lower()
    return "t5" if "t5" in model_name else "decoder"


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(Path(__file__).resolve().parent / "config.ini")
    model_id = _get_model_id(config)
    dataset_name = _get_dataset_name(config)

    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    relation_id = config['DATAPREPARATION'].get('relation_id') or config['DATAPREPARATION'].get('relation_ids')
    all_train_data = config['PROMPTPREPARATION']['all_train_data']
    all_tasks = config['PROMPTPREPARATION']['all_tasks']
    out_folder = config['PROMPTPREPARATION']['out_folder']
    relation_num = int(config['DATAPREPARATION']['relation_types'])
    memory_size = int(config['MODEL']['memory_size'])
    ## 1. Data preparation and preprocessing
    if dataset_name == 'fewrel':

        data_preparation_fewrel.main(input_file, output_file, relation_id)
        instruction_ft_data_same_setting_fewrel.main(all_train_data, all_tasks, out_folder, relation_id)
        
    elif dataset_name == 'tacred':
        
        all_test_data = config['PROMPTPREPARATION']['all_test_data']
        all_dev_data = config['PROMPTPREPARATION']['all_dev_data']
        data_preparation_tacred.main(input_file, output_file)
        instruction_ft_data_same_setting_tacred.main(all_train_data, all_dev_data, all_test_data, all_tasks, out_folder)
    else:
        print('Dataset not supported')

    ## 2. Model Training, Evaluation and Prediction
    if model_id == 't5':
        trainer_t5.trainer(config, memory_size, relation_num)
    elif model_id == 'decoder':
        trainer_decoder.trainer(config, memory_size, relation_num)
    else:
        print('Model not supported')
  

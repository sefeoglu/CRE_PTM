import sys
import os
import configparser
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.data import data_preparation_fewrel, data_preparation_tacred, instruction_ft_data_same_setting_fewrel, instruction_ft_data_same_setting_tacred

from src.CRE import trainer_t5, trainer_decoder


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    print(config['MODEL']['model_id'])
    input_file = config['DATAPREPARATION']['input_file']
    output_file = config['DATAPREPARATION']['output_file']
    relation_id = config['DATAPREPARATION']['relation_id']
    all_train_data = config['PROMPTPREPARATION']['all_train_data']
    all_tasks = config['PROMPTPREPARATION']['all_tasks']
    out_folder = config['PROMPTPREPARATION']['out_folder']
    ## 1. Data preparation and preprocessing
    if config['DATA']['dataset'] == 'fewrel':

        data_preparation_fewrel.main(input_file, output_file, relation_id)
        instruction_ft_data_same_setting_fewrelmain(all_train_data, all_tasks, out_folder, relation_id)
    elif config['DATA']['dataset'] == 'tacred':
        all_test_data = config['PROMPTPREPARATION']['all_test_data']
        all_dev_data = config['PROMPTPREPARATION']['all_dev_data']
        data_preparation_tacred.main(input_file, output_file)
        instruction_ft_data_same_setting_tacred.main(all_train_data, all_tasks, out_folder)
    else:
        print('Dataset not supported')

    ## 2. Model Training, Evaluation and Prediction
    if config['MODEL']['model_id'] == 't5':
        trainer_t5.trainer(config, memory_size=10)
    elif config['MODEL']['model_id'] == 'decoder':
        trainer_decoder.trainer(config,memory_size=10)
    else:
        print('Model not supported')
  


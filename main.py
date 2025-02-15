import sys
import os
import configparser
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.data import data_preparation_fewrel, data_preparation_tacred, instruction_ft_data_same_setting_fewrel, instruction_ft_data_same_setting_tacred

from src.evaluation import evaluate_model, write_json
from sr.CRE import trainer_t5, trainer_decoder


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    print(config['MODEL']['model_id'])
    
    ## 1. Data preparation and preprocessing
    if config['DATA']['dataset'] == 'fewrel':
        data_preparation_fewrel.main()
        instruction_ft_data_same_setting_fewrel.main()
    elif config['DATA']['dataset'] == 'tacred':
        data_preparation_tacred.main()
        instruction_ft_data_same_setting_tacred.main()
    else:
        print('Dataset not supported')

    ## 2. Model Training, Evaluation and Prediction
    if config['MODEL']['model_id'] == 't5':
        trainer_t5.main()
    elif config['MODEL']['model_id'] == 'decoder':
        trainer_decoder.main()
    else:
        print('Model not supported')
  


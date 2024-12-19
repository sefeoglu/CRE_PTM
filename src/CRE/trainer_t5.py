

import json
import evaluate
import nltk, torch
from datetime import datetime
import numpy as np
from memory.kmeans_sampleselection import select_samples
from continuous_fine_tuning_t5 import set_tokenizer, main

from datasets import load_dataset


nltk.download("punkt")
metric = evaluate.load("rouge")
from evaluation import evaluate_model, write_json, read_json

def trainer(m):

    for experiment_id in range(1,6):

        print("Experiment: {0}".format(experiment_id))

        dataset_path = 'fewrel/train/run_{0}/task1/'.format(experiment_id)
        tasks_path = "relations/run_{0}/task1.json".format(experiment_id)
        model_id="google/flan-t5-base"
        task_id = "task1"
        targets_labels = []
        max_source_length = None
        max_target_length = None
        tokenizer = set_tokenizer()



        metric = evaluate.load("rouge")
        start_time = datetime.now()
        model, tokenizer, trainer = main(model_id, dataset_path, tasks_path, task_id, False)
        end_time = datetime.now()
        train_time = 'Base Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
        logs += train_time

        model.save_pretrained("KMmeans_CRE_fewrel_{0}/task_memory_5_1/".format(experiment_id), from_pt=True)
        ## evaluate model
        evaluate_model(experiment_id, task_id, model, tokenizer, current_task=True)
        if m >0:
            train_data_path = dataset_path+"train_1.json"
            all_selected_samples = select_samples(model, tokenizer, m, train_data_path, tasks_path)

            for k in range(2, 11):
                outpath_selected_samples = 'fewrel/train/run_{0}/task_memory_{1}/train_2.json'.format(experiment_id,k)
                write_json(all_selected_samples, outpath_selected_samples)

        for i in range(1, 10):
            targets_labels = []
            max_source_length = None
            max_target_length = None
            tokenizer = set_tokenizer()
            metric = evaluate.load("rouge")

            dataset_path = 'fewrel/train/run_{0}/task{1}/'.format(experiment_id,i+1)
            tasks_path = "relations/run_{0}/task{1}.json".format(experiment_id,i+1)

            base_model_id="KMmeans_CRE_fewrel_{0}/task_memory_5_{1}/".format(experiment_id, i)
            task_id = "task{0}".format(i+1)

            print(base_model_id)
            start_time = datetime.now()
            model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
            end_time = datetime.now()
            train_time = 'Base Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
            logs += train_time
            model.save_pretrained("KMmeans_CRE_fewrel_{0}/task_memory_5_{1}/".format(experiment_id, i+1), from_pt=True)
            #evaluate model
            evaluate_model(experiment_id, task_id, model, tokenizer, current_task=True)
            write_json(logs, "drive/MyDrive/ESWC-figures/fewrel/m_15/KMmeans_CRE_fewrel_{0}/logs.txt".format(experiment_id))
            if m > 0:
                if i <9:
                    train_data_path = dataset_path+"train_1.json"
                    all_selected_samples = select_samples(model, tokenizer,m, train_data_path, tasks_path)
                    for j in range(i+2, 11):
                        outpath_selected_samples = 'fewrel/train/run_{0}/task_memory_{1}/train_{2}.json'.format(experiment_id, j, i+2)
                        write_json(all_selected_samples, outpath_selected_samples)

                ########################### Memory Train ######################################
                targets_labels = []
                max_source_length = None
                max_target_length = None
                tokenizer = set_tokenizer()
                metric = evaluate.load("rouge")

                dataset_path = 'fewrel/train/run_{0}/task_memory_{1}/'.format(experiment_id,i+1)
                tasks_path = "relations/run_{0}/task{1}.json".format(experiment_id,i+1)

                base_model_id = "KMmeans_CRE_fewrel_{0}/task_memory_5_{1}/".format(experiment_id, i+1)
                task_id = "task{0}".format(i+1)

                print(base_model_id)
                start_time = datetime.now()
                model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
                end_time = datetime.now()

                train_time = 'Memory Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
                logs += train_time
                model.save_pretrained("KMmeans_CRE_fewrel_{0}/task_memory_5_{1}/".format(experiment_id, i+1), from_pt=True)
                ### evaluate model
                evaluate_model(experiment_id, i+1, model, tokenizer, current_task=False)
                write_json(logs, "drive/MyDrive/ESWC-figures/fewrel/m_15/KMmeans_CRE_fewrel_{0}/logs.txt".format(experiment_id))

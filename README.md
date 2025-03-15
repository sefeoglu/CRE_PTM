# Evaluating Large Language Models for Continual Relation Extraction in Incremental Task Learning


The results and datasets used in this work are available at [drive](https://drive.google.com/drive/folders/1ev9EBUaDNjTfeIPhUNcFEXthUQ3kAFVZ?usp=drive_link)
![CRE](https://github.com/sefeoglu/CRE_PTM/blob/master/doc/cre.png)
**Note** Upon acceptance, trained models will be public on HuggingFace.


## Folder Structure
```xml
.
├── LICENSE
├── README.md
├── config.ini
├── data                            -> settings and data split setting here for tacred and fewrel like relation types per task
├── doc                             -> figures
├── main.py
├── requirements.txt                -> dependecies like libraries
└── src
    ├── CRE                         -> continual training of Flan T5 Base, Llama2 and Mistral
    ├── analysis_viz                -> Visualization like logs and  section 4 figures.
    ├── clean                       -> cleaning of results of llama and mistral from instructions.
    ├── data_preparetation          -> prompt dataset generation
    ├── metrics                     -> bwt, whole and average accuracy calculation
    ├── utils.py                    -> read and write
    └── zero_shot_prompting         -> ablation study, but not in the paper.
````
        
## How it works
Setup configuration in `config.ini` according to your needs before starting running experiments.


1.) Prepare datasets:

**TACRED**:
* This command with convert data row to (sentence, subject, object, object_type and subject_type)
````bash
$ python src/data/data_prepare_tacred.py
````
* Split datasets according to setting Cui et al. 2021
````bash
$ python src/data/instruction_ft_data_same_setting_tacred.py
````
**FewRel**
* Same steps with TACRED
````bash
$ python src/data/data_preparation_fewrel.py
````
* split
````bash
$ python src/data/instruction_ft_data_same_setting_fewrel.py
```` 
2.)Trainer

 * Decoder only models(Llama2-7B-chat-hf and Mistral-Instruct-7B-v2.0)
````bash
$ python python src/CRE/trainer_decoder.py
````
 * Encoder-Decoder model(Flan T5-Base)
````bash
$ python src/CRE/trainer_t5.py
````
3.) Clean decoder-only models results from explainations

````bash
$ python src/clean/clean_results.py
````
4.) Metrics


Average and Whole Accuracy Metrics
````bash
$ python src/metrics/cl_metrics.py

````
Backward Knowledge Transfer Computation
````bash
$ python src/metrics/bwt.py
````

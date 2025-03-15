# Evaluating Large Language Models for Continual Relation Extraction in Incremental Task Learning


The results and datasets used in this work are available at [drive](https://drive.google.com/drive/folders/1ev9EBUaDNjTfeIPhUNcFEXthUQ3kAFVZ?usp=drive_link)
![CRE](https://github.com/sefeoglu/CRE_PTM/blob/master/doc/cre.png)
**Note** Upon acceptance, trained models will be shared on Huggingface.


## Folder Structure
```xml
.
├── LICENSE
├── README.md
├── config.ini
├── data
│   ├── fewrel                      -> settings and data split setting here
│   └── tacred                      -> settings and data split setting here
├── doc                             -> Design figure
├── requirements.txt                -> Libraries
├── resulting_metrics               -> resulting metrics will be saved here (will be shared on huggingface)
└── src
    ├── CRE                         -> models, trainer, evaluation, kmeans
    ├── clean                       -> cleaning mistral and llama results from explaination
    ├── data_preparation            -> prompt dataset preparation
    ├── metrics                     -> bwt, resulting model metrics(acc and whole)
    ├── viz                         -> log viz and pca in notebooks
    ├── utils.py
    └── zero_shot_prompting         -> ablation study
````
        
## How works
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
$ python src/clean/clean_mistral_results.py
$ python src/clean/clean_llama_results.py
````
4.) Metrics
````bash
$ python src/metrics/resulting_model_metrics.py
$ python src/metrics/bwt.py
````

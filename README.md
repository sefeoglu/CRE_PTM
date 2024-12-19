# Continual Relation Extraction with Pretrained Large Language Models (CRE-PLM)
Continual Relation Extraction Utilizing Pretrained Large Language Model
**WIP!** (for refactoring and cleaning for hardcoding)
**Note** Upon acceptance, the prediction results, and metrics results and trained models will be shared on Huggingface.

## Method
![Method](https://github.com/sefeoglu/CRE_PTM/blob/master/doc/CRE_PLM.png)

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
    ├── data-preparetation          -> prompt dataset preparation
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
$ python src/data-preparetation/data_prepare_tacred.py
````
* Split datasets according to setting Cui et al. 2021
````bash
$ python src/data-preparetation/instruction_ft_data_same_setting_tacred.py
````
**FewRel**
* Same steps with TACRED
````bash
$ python src/data-preparetation/data_preparation_fewrel.py
````
* split
````bash
$ python src/data-preparetation/instruction_ft_data_same_setting_fewrel.py
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
Note: The seen and current taks accuracies will be computed over training with ```src/CRE/evaluation.py``` 
## Results
### 1.)  Mean Seen Accuracy (%) for Task-Incremental Training

**Table:** Mean seen accuracy (%) of task-incremental training on the TACRED and FewRel datasets over ten tasks and five runs. The results are given for memory size 10 as evaluated in previous works. Highest accuracies are highlighted in blue.

| **Dataset** | **Method**                                      | **1**    | **2**    | **3**    | **4**    | **5**    | **6**    | **7**    | **8**    | **9**    | **10**   |
|-------------|-------------------------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| **TACRED**  | **EMAR**~[Han et al., 2020]                     | 73.6     | 57.0     | 48.3     | 42.3     | 37.7     | 34.0     | 32.6     | 30.0     | 27.6     | 25.1     |
|             | **EA-EMR**~[Wang et al., 2019]                  | 47.1     | 40.1     | 38.3     | 29.9     | 28.4     | 27.3     | 26.9     | 25.8     | 22.9     | 19.8     |
|             | **CML**~[Wu et al., 2021]                       | 57.2     | 51.4     | 41.3     | 39.3     | 35.9     | 28.9     | 27.3     | 26.9     | 24.8     | 23.4     |
|             | **EMAR-BERT**~[Han et al., 2020]                | 96.6     | 85.7     | 81.0     | 78.6     | 73.9     | 72.3     | 71.7     | 72.2     | 72.6     | 71.0     |
|             | **RP-CRE**~[Cui et al., 2021]                   | 97.6     | 90.6     | 86.1     | 82.4     | 79.8     | 77.2     | 75.1     | 73.7     | 72.4     | 72.4     |
|             | **CRL**~[Zhao et al., 2022]                     | 97.7     | 93.2     | 89.8     | 84.7     | 84.1     | 81.3     | 80.2     | 79.1     | 79.0     | 78.0     |
|             | **KIP-Framework**~[Zhang, 2022]                 | **98.3** | 95.0     | 90.8     | 87.5     | 85.3     | 84.3     | 82.1     | 80.2     | 79.6     | 78.6     |
|             | **CREST**~[Le & Nguyen, 2024]                   | 97.3     | 91.4     | 82.3     | 82.5     | 79.2     | 75.8     | 78.8     | 77.4     | 78.6     | 79.4     |
|             | **Ours with Flan-T5 Base**                      | 96.0     | **96.2** | 95.7     | **96.0** | 95.7     | 95.4     | 96.0     | 96.0     | **96.3** | 95.8     |
|             | &nbsp;&nbsp;&nbsp;&nbsp;**with Mistral-7B-Instruct-v2.0** | 95.0     | 94.8     | **96.4** | 96.0     | **96.6** | **97.0** | **96.8** | **96.9** | 95.8     | **96.9** |
|             | &nbsp;&nbsp;&nbsp;&nbsp;**with Llama2-7B-chat-hf**      | 55.5     | 54.7     | 43.8     | 43.4     | 51.4     | 71.0     | 61.1     | 72.6     | 73.6     | 69.6     |
| **FewRel**  | **EA-EMR**~[Wang et al., 2019]                  | 88.5     | 69.0     | 59.1     | 54.2     | 47.8     | 46.1     | 43.1     | 40.7     | 38.6     | 35.1     |
|             | **EMAR**~[Han et al., 2020]                     | 88.5     | 73.2     | 66.6     | 63.8     | 55.8     | 54.3     | 52.9     | 50.9     | 48.8     | 46.3     |
|             | **CML**~[Wu et al., 2021]                       | 91.2     | 74.8     | 68.2     | 58.2     | 53.7     | 50.4     | 47.8     | 44.4     | 43.1     | 39.7     |
|             | **EMAR-BERT**                                   | **98.8** | 89.1     | 89.5     | 85.7     | 83.6     | 84.8     | 79.3     | 80.0     | 77.1     | 73.8     |
|             | **RP-CRE**~[Cui et al., 2021]                   | 97.9     | 92.7     | 91.6     | 89.2     | 88.4     | 86.8     | 85.1     | 84.1     | 82.2     | 81.5     |
|             | **CRL**~[Zhao et al., 2022]                     | 98.2     | 94.6     | 92.5     | 90.5     | 89.4     | 87.9     | 86.9     | 85.6     | 84.5     | 83.1     |
|             | **KIP-Framework**~[Zhang, 2022]                 | 98.4     | 93.5     | 92.0     | 91.2     | 90.0     | 88.2     | 86.9     | 85.6     | 84.1     | 82.5     |
|             | **CREST**~[Le & Nguyen, 2024]                   | 98.7     | 93.6     | **93.8** | **92.3** | **91.0** | **89.9** | **87.6** | **86.7** | **86.0** | **84.8** |
|             | **Ours with Flan-T5 Base**                      | 97.3     | **94.4** | 91.9     | 90.2     | 87.7     | 85.7     | 84.1     | 79.8     | 77.1     | 70.0     |


### 2.)  Mean Average Accuracy (a) and Whole Accuracy (w) (%) on the TACRED and FewRel Datasets

Mean Average Accuracy (a) and Whole Accuracy (w) (%) across 5 runs on the TACRED and FewRel datasets.  
The second-best results are highlighted in green, while the best results are highlighted in blue.

| **Method**                         | **TACRED w** | **TACRED a** | **FewRel w**         | **FewRel a**         |
|------------------------------------|--------------|--------------|----------------------|----------------------|
| **EMR**                            | 21.8         | 26.5         | 42.0                | 54.1                |
| **EA-EMR**                         | 23.0         | 30.0         | 49.0                | 61.2                |
| **EMAR**                           | 31.0         | 36.3         | 53.8                | 68.1                |
| **CML**                            | 43.7         | 45.3         | --                  | --                  |
| **KIP-Framework** [Zhang et al., 2022] | 91.1         | 91.6         | **<span style="color:blue;">96.3</span>** | **<span style="color:blue;">96.6</span>** |
| **Ours with Flan-T5**              | *<span style="color:green;">95.76</span>* | *<span style="color:green;">95.78</span>* | 70.33               | 70.33               |
| &nbsp;&nbsp;&nbsp;**with Mistral** | **<span style="color:blue;">96.89</span>** | **<span style="color:blue;">96.76</span>** | --                  | --                  |
| &nbsp;&nbsp;&nbsp;**with Llama2**  | 68.01        | 67.58        | *<span style="color:green;">86.30</span>* | *<span style="color:green;">86.30</span>* |

Note: "--" indicates results are not available.

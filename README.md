# Large Language Models for Continual Relation Extraction
The results of this work have been submitted to Springer Nature Machine Learning Journal!
It is under revision !


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
├── results                         -> results for TACRED with Flan-T5 and All Results for FewRel
├── logs                            -> time cost logs for each experiment and FewRel's in side of FewRel results
├── main.py
├── requirements.txt                -> dependecies like libraries
└── src
    ├── CRE                         -> continual training of Flan T5 Base, Llama2 and Mistral
    ├── analysis_viz                -> Visualization like logs and  section 4 figures.
    ├── clean                       -> cleaning of results of llama and mistral from explainations and instructions.
    ├── data_preparetation          -> prompt dataset generation
    ├── metrics                     -> bwt, whole and average accuracy calculation
    ├── utils.py                    -> read and write
    └── zero_shot_prompting         -> ablation study, but not in the paper.
````
        
## How it works
Setup configuration in `config.ini` according to your needs before starting running experiments.
```bash
$ python main.py
```
or 
follow the steps below.


1.) Prepare datasets:

**TACRED**:
* This command with convert data row to (sentence, subject, object, object_type and subject_type)
````bash
$ python src/data_preparetation/data_prepare_tacred.py
````
* Split datasets according to setting Cui et al. 2021
````bash
$ python src/data_preparetation/instruction_ft_data_same_setting_tacred.py
````
**FewRel**
* Same steps with TACRED
````bash
$ python src/data_preparetation/data_preparation_fewrel.py
````
* split
````bash
$ python src/data_preparetation/instruction_ft_data_same_setting_fewrel.py
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
$ python src/clean/clean_decoder_results.py
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
### References
```
@inproceedings{cui-etal-2021-refining,
  title     = {{R}efining {S}ample {E}mbeddings with {R}elation {P}rototypes to {E}nhance {C}ontinual {R}elation {E}xtraction},
  author    = {Cui, Li and Yang, Deqing and Yu, Jiaxin and Hu, Chengwei and Cheng, Jiayang and Yi, Jingjie and Xiao, Yanghua},
  editor    = {Zong, Chengqing and Xia, Fei and Li, Wenjie and Navigli, Roberto},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  month     = {8},
  year      = {2021},
  address   = {Online},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2021.acl-long.20},
  doi       = {10.18653/v1/2021.acl-long.20},
  pages     = {232--243}
}
```

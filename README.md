`eda.ipynb` - basic data checks and pipleine drafts
`bert.ipynb` - model training and all transformations documented in appropriate format


### Description Classification to Branch 

For baseline used LogReg+TF-IDF from scikit-learn. Default parameters.

Dataset Rows: ~633k

Base score:
    Accuracy: 0.6348
    Macro F1: 0.5961

For BERT-distil from training on 33/50k batches:
    Accuracy: 0.6997
    Macro F1: 0.6699


Curerently we see that due to having 25 classes it's hard to make very high metrics on f1-macro, bc we have few training examples for some of classes.
Nevertheless for categories with much training data as "IT" our model classify with higher confidence. 
At the moment our distil-bert doesn't show much higher results as basic log-reg, but if we will have less classes we might got much higher gap in classification metrics for these models. 


### NER skill extraction - tokens sequence labeling
Data : https://huggingface.co/datasets/jjzha/skillspan

For baseline used fratures-CRF from sckikit_crf. Parameters algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True
Dataset Rows ~ 4k

Base score:
Validaiton data : Precision: 0.3375  Recall: 0.1000  F1: 0.1543
Test data : Precision: 0.3607  Recall: 0.1009  F1: 0.1577  

Bert-CRF:


### Notes for development (enternal doc):
NER datasets sources:
1. https://huggingface.co/datasets?sort=trending&search=skillspan
2. https://huggingface.co/datasets?sort=trending&search=job+skill
3. https://github.com/kris927b/SkillSpan?tab=readme-ov-file or https://aclanthology.org/2022.naacl-main.366/


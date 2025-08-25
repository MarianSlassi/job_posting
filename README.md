`eda.ipynb` - basic data checks and pipleine drafts
`bert.ipynb` - model training and all transformations documented in appropriate format

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




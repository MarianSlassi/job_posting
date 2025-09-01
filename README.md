`eda.ipynb` - basic data checks and pipleine drafts
`bert.ipynb` - model training and all transformations documented in appropriate format


## Description Classification to industry/job Branch 
Data:
https://huggingface.co/datasets/2024-mcm-everitt-ryan/job-postings-english-clean
<br>

For baseline used LogReg+TF-IDF from scikit-learn withdDefault parameters.

Dataset Rows: ~633k

### Base score:
    Accuracy: 0.6348
    Macro F1: 0.5961

### For BERT-distil from training on 33/50k batches:
    Accuracy: 0.6997
    Macro F1: 0.6699


Result: Curerently we see that due to having 25 classes it's hard to make very high metrics on f1-macro, bc we have few training examples for some of classes.
Nevertheless for categories with much training data as "IT" our model classify with higher confidence. 
At the moment our distil-bert doesn't show much higher results as basic log-reg, but if we will have less classes we might got much higher gap in classification metrics for these models. 


## NER skill extraction - tokens sequence labeling
Data : https://huggingface.co/datasets/jjzha/skillspan

For baseline used features-CRF from sckikit_crf. Parameters algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True
Dataset Rows ~ 4k

### Base score features-CRF:

    Validaiton data: 
        Precision: 0.3375
        Recall: 0.1000
        F1: 0.1543
    
    Test data: 
        Precision: 0.3607
        Recall: 0.1009
        F1: 0.1577  

### Bert-MLP:
    Validation data:
        precision: 0.4822
        recall: 0.5308
        f1-score:0.5053 

    Test data: 
        precision: 0.4896
        recall: 0.4734
        f1-score: 0.4813 


Result: BERT-MLP shows high score compare to features-CRF. But entity boundaries are frequently misaligned, with spans predicted partially or with extra tokens.
There are occasional false positives on generic words like skills or experience.
Short single-word skills (“Consulting”, “communication”) are predicted worse than multi-word terms.
### Notes for development (enternal doc):
NER datasets sources:
1. https://huggingface.co/datasets?sort=trending&search=skillspan
2. https://huggingface.co/datasets?sort=trending&search=job+skill
3. https://github.com/kris927b/SkillSpan?tab=readme-ov-file or https://aclanthology.org/2022.naacl-main.366/



### Lock

console if \__main__ initialised : uv run python -m src.main_test
if __name__ == '__main__':
    uvicorn.run("src.main_test:app", host="127.0.0.1", port=8000, reload=True)

if just no code you can run with 
 uv run uvicorn src.main_test:app --reload



### How to use skills_model
from pathlib import Path
from skill_extractor import SkillExtractor  

train_path = Path("data/skill_conll/train.conll")
val_path   = Path("data/skill_conll/validation.conll")
test_path  = Path("data/skill_conll/test.conll")
se = SkillExtractor(model_name="bert-base-cased", max_length=256)
se.prepare_from_conll(train_path, val_path, test_path)

out_dir = Path("models/bert-skill-ner")
se.fit(output_dir=out_dir, early_stopping_patience=3)
se.evaluate("validation")
se.evaluate("test")

save_dir = out_dir / "final"
se.save(save_dir)

se2 = SkillExtractor.load(save_dir)

text = "Senior Python Developer with FastAPI, Docker and a bit of Kubernetes; strong SQL."
skills = se2.predict(text)               
print(skills)

skills_all = se2.predict(text, unique=False)
print(skills_all)


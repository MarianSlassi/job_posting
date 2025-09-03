# NLP WEB-API >>  Classification & Skills Extraction with HuggingFace + FastAPI


## Description: Classification to Industry/Job Branch
**Data:** [Job Postings Dataset](https://huggingface.co/datasets/2024-mcm-everitt-ryan/job-postings-english-clean)

For the baseline, we used **LogReg+TF-IDF** from scikit-learn with default parameters.

**Dataset Rows:** ~633k

### Base score:
- **Accuracy:** 0.6348
- **Macro F1:** 0.5961

### For BERT-distil (training on 33/50k batches):
- **Accuracy:** 0.6997
- **Macro F1:** 0.6699

**Result:** Currently, we see that due to having 25 classes, it's hard to achieve very high metrics on F1-macro because we have few training examples for some classes. Nevertheless, for categories with much training data, such as "IT", our model classifies with higher confidence. At the moment, our distil-bert doesn't show much higher results than the basic LogReg, but if we had fewer classes, we might get a much higher gap in classification metrics for these models.

## NER Skill Extraction - Tokens Sequence Labeling
**Data:** [SkillSpan Dataset](https://huggingface.co/datasets/jjzha/skillspan)

For the baseline, we used **features-CRF** from `scikit_crf`. Parameters:
- `algorithm='lbfgs'`
- `c1=0.1`
- `c2=0.1`
- `max_iterations=200`
- `all_possible_transitions=True`

**Dataset Rows:** ~4k

### Base score features-CRF:
**Validation data:**
- **Precision:** 0.3375
- **Recall:** 0.1000
- **F1:** 0.1543

**Test data:**
- **Precision:** 0.3607
- **Recall:** 0.1009
- **F1:** 0.1577

### Bert-MLP:
**Validation data:**
- **Precision:** 0.4822
- **Recall:** 0.5308
- **F1-score:** 0.5053

**Test data:**
- **Precision:** 0.4896
- **Recall:** 0.4734
- **F1-score:** 0.4813

**Result:** BERT-MLP shows higher scores compared to features-CRF. However, entity boundaries are frequently misaligned, with spans predicted partially or with extra tokens. There are occasional false positives on generic words like "skills" or "experience". Short single-word skills (e.g., "Consulting", "communication") are predicted worse than multi-word terms.

## 🦎 Notes for Development (Internal Doc):
**NER Datasets Sources:**
1. [Huggingface - SkillSpan](https://huggingface.co/datasets?sort=trending&search=skillspan)
2. [Huggingface - Job Skills](https://huggingface.co/datasets?sort=trending&search=job+skill)
3. [SkillSpan GitHub](https://github.com/kris927b/SkillSpan?tab=readme-ov-file) or [ACL Anthology](https://aclanthology.org/2022.naacl-main.366/)



### Uvicorn

    console if \__main__ initialised : uv run python -m src.app.main_test
    if __name__ == '__main__':
        uvicorn.run("src.app.main_test:app", host="127.0.0.1", port=8000, reload=True)

    if just no code you can run with 
    uv run uvicorn src.main_test:app --reload

###  How to use Description classification model: 
    from src.model import ClassifierModel 
    finetuned_dir = "../models/checkpoint-33000"

    clf = ClassifierModel(finetuned_dir, finetuned=True, max_length=256)

    text = "Senior Python developer needed in Amsterdam, experience with NLP required."
    pred = clf.predict(text, return_probas=True)
    print(pred)

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


## API testing
### post: /extract_skills
    {"text": "You will work as part of our Agile hybrid DevOps teams to help develop/configure new tools roll out environments and automate processes using a variety of tools and techniques. You have possibility to sell, give and evaluate sollutions. Plus our machine learning model need you help and proffesional consultations. Give it as much functionality and possibilities as possible"}

### post: /classify
    {"text": "Python JavaScript Developer. With possibility to conduct smooth sales"}


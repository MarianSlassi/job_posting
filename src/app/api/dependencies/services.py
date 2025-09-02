from src.core.classifiers.classification_model import ClassifierModel
from src.core.ner.skills_model import SkillExtractor

class ClassifierService():
    """
    Wraps both classical (TF-IDF + Logistic Regression) and modern (BERT) pipelines.
    Replace the dummy logic with real .predict() using your trained artifacts.
    """
    def __init__(self, bert_model: ClassifierModel, logger) -> None:
        self.bert_model= bert_model
        logger.info("ClassifierService initialized.")

    def predict(self, text: str) -> dict:
        if not self.bert_model:
            raise RuntimeError("BERT classifier is not loaded.")

        return self.bert_model.predict(text)


class SkillExtractorService:

    def __init__(self, ner_model: SkillExtractor, logger) -> None:
        self.ner_model = ner_model


    def predict(self, text: str):
        return self.ner_model.predict(text)
# -----------------------------------------------------------------------------
# Model service layer 
# -----------------------------------------------------------------------------
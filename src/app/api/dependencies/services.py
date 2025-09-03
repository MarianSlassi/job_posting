from src.app.core.classifiers.classification_model import ClassifierModel
from src.app.core.ner.skills_model import SkillExtractor
from src.app.logs import get_custom_logger

class ClassifierService():

    def __init__(self, bert_model: ClassifierModel) -> None:
        self.bert_model= bert_model
        
        self.classifier_logger = get_custom_logger(log_file= 'classifier', name= 'classifier_service')
        self.classifier_logger.info("ClassifierService initialized.")

    def predict(self, text: str) -> dict:
        if not self.bert_model:
            raise RuntimeError("BERT classifier is not loaded.")

        return self.bert_model.predict(text)


class SkillExtractorService:

    def __init__(self, ner_model: SkillExtractor) -> None:
        self.ner_model = ner_model
        self.ner_logger = get_custom_logger(log_file= 'ner', name= 'ner_service')
        self.ner_logger.info("Extraction Service initialized.")


    def predict(self, text: str):
        if not self.ner_model:
            raise RuntimeError("BERT extractor is not loaded.")
        return self.ner_model.predict(text)

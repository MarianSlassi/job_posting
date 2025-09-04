from src.app.core.classifiers.classification_model import ClassifierModel
from src.app.core.ner.skills_model import SkillExtractor

class ClassifierService():

    def __init__(self, classifier_model: ClassifierModel) -> None:
        self.classifier_model= classifier_model

    def predict(self, text: str) -> dict:
        if not self.classifier_model:
            raise RuntimeError("Classifier model class is not loaded.")
        return self.classifier_model.predict(text)


class SkillExtractorService:

    def __init__(self, extractor: SkillExtractor) -> None:
        self.extractor = extractor

    def predict(self, text: str):
        if not self.extractor:
            raise RuntimeError("Extractor model class is not loaded.")
        return self.extractor.predict(text)

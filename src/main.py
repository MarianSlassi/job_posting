# app.py
from typing import List, Optional, Literal, Any
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from contextlib import asynccontextmanager



from src.config import Config
from src.app.logs import get_custom_logger

from src.app.api.schemas import RequestText, ClassifyTextResponse, SkillExtractionItem, SkillExtractionResponse

from src.core.classifiers.classification_model import ClassifierModel
from src.core.ner.skills_model import SkillExtractor

from src.app.api.dependencies import ClassifierService, SkillExtractorService
from src.app.api.dependencies import get_extractor, get_classifier, get_config, get_logger


from src.app.api.routers.classify import classify_router
from src.app.api.routers.health import health_router
from src.app.api.routers.extract_skill import extract_skills_router

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
# logger = logging.getLogger("job-nlp-api")
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
# logger.addHandler(handler)


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading config...")
    app.state.config = Config()
    app.state.logger = get_custom_logger(config = app.state.config, log_file= 'app', name= 'job-nlp-api')
    print("Loading models...")
    app.state.classifier = ClassifierService(logger = app.state.logger, bert_model = ClassifierModel(model_path=str(app.state.config.get('classification_model'))))
    print("Classifier model has been load...")
    app.state.extractor = SkillExtractorService(logger = app.state.logger, ner_model = SkillExtractor.load(load_dir=app.state.config.get('ner_model')))
    print("SkillExtractor model has been load...")

    yield  

    print("Shutting down...")

# -----------------------------------------------------------------------------
# Application factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="TalentStream NLP API",
        version="0.1.0",
        description="FastAPI service for job classification and skill extraction.",
        lifespan = lifespan
    )
    app.include_router(classify_router)
    app.include_router(health_router)
    app.include_router(extract_skills_router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)

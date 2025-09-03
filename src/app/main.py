from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.app.config import Config
from src.app.logs import get_custom_logger

from src.app.core.classifiers.classification_model import ClassifierModel
from src.app.core.ner.skills_model import SkillExtractor

from src.app.api.dependencies import ClassifierService, SkillExtractorService

from src.app.api.routers.classify import classify_router
from src.app.api.routers.health import health_router
from src.app.api.routers.extract_skill import extract_skills_router


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()
    lifespan_logger = get_custom_logger(log_file= 'lifespan', name= 'lifespan')

    lifespan_logger.info("Loading models...")
    app.state.classifier = ClassifierService(bert_model = ClassifierModel(model_path=str(config.get('classification_model'))))
    lifespan_logger.info("Classifier model has been load...")
    app.state.extractor = SkillExtractorService(ner_model = SkillExtractor.load(load_dir=config.get('ner_model')))
    lifespan_logger.info("SkillExtractor model has been load...")

    yield  

    lifespan_logger.info("Shutting down...")

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
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=True)

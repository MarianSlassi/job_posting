from typing import Any
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Request
from src.app.api.schemas import RequestText, ClassifyTextResponse
from src.app.api.dependencies.client import get_classifier
from src.app.api.responses.classify import classify_responses

classify_router = APIRouter()


@classify_router.post("/classify", response_model=ClassifyTextResponse, responses = classify_responses)  
async def classify(req: RequestText, classifier = Depends(get_classifier)) -> Any:
    raw_result = classifier.predict(text=req.text)
    return ClassifyTextResponse(
        category=raw_result["label"],  
        score=raw_result["score"]
    )
        



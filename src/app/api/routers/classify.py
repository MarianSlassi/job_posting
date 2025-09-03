from typing import Any
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Request
from src.app.api.schemas import RequestText, ClassifyTextResponse
from src.app.api.dependencies.client import get_classifier
from src.app.api.responses.classify import classify_responses
from src.app.logs import get_custom_logger

classify_router = APIRouter()


@classify_router.post("/classify", response_model=ClassifyTextResponse, responses = classify_responses)  
async def classify(req: RequestText, classifier = Depends(get_classifier), request: Request = None) -> Any:
    logger = get_custom_logger(log_file= 'classifier_endpoint', name= 'classifier_endpoint')
    try:
        raw_result = classifier.predict(text=req.text)
        logger.info(f"Processed classify request: {raw_result}")
        if len(raw_result) == 0:
            return ClassifyTextResponse(category='None', score='0')
        service_response = ClassifyTextResponse(
            category=raw_result["label"],  
            score=raw_result["score"]
        )
        if service_response:
            logger.info(f'Classification service has processed request and gave response: {service_response}')
        return service_response
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

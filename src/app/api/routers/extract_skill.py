from typing import Any
from fastapi import HTTPException, APIRouter, Depends
from src.app.api.schemas import RequestText, SkillExtractionResponse, SkillExtractionItem
from src.app.api.dependencies import get_extractor
from src.app.api.responses.extract_skills import extract_skills_responses
from src.app.logs import get_custom_logger

extract_skills_router = APIRouter()

@extract_skills_router.post("/extract_skills", response_model= SkillExtractionResponse, responses = extract_skills_responses) # 
async def extract_skills(req: RequestText, extractor = Depends(get_extractor)) -> Any:
    logger = get_custom_logger(log_file= 'extractor_endpoint', name= 'extractor_endpoint')
    logger.info(f"Received extract_skills request: {req}")
    try:
        raw_result = extractor.predict(text = req.text)
        logger.info(f"Processed extract_skills request: {raw_result}")
        if len(raw_result) == 0:
            return SkillExtractionResponse(skills=[SkillExtractionItem(text="No Skills found")])
        service_response = SkillExtractionResponse( 
            skills=[SkillExtractionItem(text=skill) for skill in raw_result]
        )
        if service_response:
            logger.info(f'Skills Extraction Service has processed request and gave response: {service_response}')
        return service_response
    except Exception as e:
        logger.error(f"Error in extract_skills endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

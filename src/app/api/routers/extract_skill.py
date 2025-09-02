from typing import Any
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Request
from src.app.api.schemas import RequestText, SkillExtractionResponse, SkillExtractionItem
from src.app.api.dependencies import get_extractor
from src.app.api.responses.extract_skills import extract_skills_responses

extract_skills_router = APIRouter()

@extract_skills_router.post("/extract_skills", response_model= SkillExtractionResponse, responses = extract_skills_responses) # 
async def extract_skills(req: RequestText, extractor = Depends(get_extractor)) -> Any:
    raw_result = extractor.predict(text = req.text)

    return SkillExtractionResponse( 
        skills=[SkillExtractionItem(text=skill) for skill in raw_result]
    )
    # return app.state.extractor.predict(text = req.text)
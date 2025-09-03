from fastapi import status
from src.app.api.schemas import SkillExtractionResponse

extract_skills_responses = {
    status.HTTP_200_OK: {
        "description": "Successful skill extraction",
        "model": SkillExtractionResponse,
        "content": {
            "application/json": {
                "example": {
                    "skills": [
                        {"text": "Python"},
                        {"text": "FastAPI"},
                        {"text": "Machine Learning"}
                    ]
                }
            }
        }
    },
    status.HTTP_400_BAD_REQUEST: {
        "description": "Bad request (e.g., missing 'text' field)"
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "description": "Validation error in input data"
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "description": "Internal server error"
    }
}

from fastapi import status
from src.app.api.schemas import ClassifyTextResponse

classify_responses = {
    status.HTTP_200_OK: {
        "description": "Successful job classification",
        "model": ClassifyTextResponse,
        "content": {
            "application/json": {
                "example": {
                    "job_title": "Software Engineer",
                    "classification": "IT",
                    "confidence": 0.97
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

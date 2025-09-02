from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class RequestText(BaseModel):
    text: str = Field(..., min_length=3, description="Job description plain text",\
                       example="You will work as part of our Agile hybrid DevOps teams to help develop/configure new tools roll out environments and automate processes using a variety of tools and techniques")


class ClassifyTextResponse(BaseModel):
    """Classification result for one item."""
    category: str = Field(..., description="category, e.g., 'Software', 'Marketing'")
    score: Optional[float] = Field(None, ge=0.0, le=1.0)

class SkillExtractionItem(BaseModel):
    """A single skill mention with character offsets."""
    text: str = Field(..., description="Continious part of job posting description classified as a skill")

class SkillExtractionResponse(BaseModel):
    skills: List[SkillExtractionItem]

# -----------------------------------------------------------------------------
# Domain schemas (Pydantic models for request/response validation)
# -----------------------------------------------------------------------------

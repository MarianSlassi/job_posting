# app.py
from typing import List, Optional, Literal, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import logging

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("job-nlp-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)

# -----------------------------------------------------------------------------
# Domain schemas (Pydantic models for request/response validation)
# -----------------------------------------------------------------------------
class JobText(BaseModel):
    """Single job description item."""
    id: Optional[str] = Field(None, description="Optional client-side identifier")
    text: str = Field(..., min_length=3, description="Job description plain text")

    @validator("text")
    def strip_text(cls, v: str) -> str:
        # Basic normalization
        return v.strip()


class ClassifyRequest(BaseModel):
    """Request schema for classification (supports batch)."""
    items: List[JobText] = Field(..., min_items=1, description="List of job descriptions to classify")
    # Optional switches to choose backend model. Replace with your own logic.
    model: Literal["tfidf_lr", "bert"] = "bert"


class ClassifyItemResult(BaseModel):
    """Classification result for one item."""
    id: Optional[str]
    level1: str = Field(..., description="Top-level category, e.g., 'Software'")
    level2: str = Field(..., description="Second-level category, e.g., 'Backend Engineer'")
    # Optionally include probabilities or scores if needed
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    """Response schema for classification."""
    model_used: str
    results: List[ClassifyItemResult]


class ExtractRequest(BaseModel):
    """Request schema for skill extraction (supports batch)."""
    items: List[JobText] = Field(..., min_items=1, description="List of job descriptions to extract skills from")
    model: Literal["crf", "bert_ner"] = "bert_ner"
    # Controls span merging / normalization options
    lower: bool = True
    unique: bool = True


class SkillSpan(BaseModel):
    """A single skill mention with character offsets."""
    text: str
    start: int = Field(..., ge=0, description="Start char index in the original text")
    end: int = Field(..., ge=0, description="End char index (exclusive)")
    label: Literal["SKILL"] = "SKILL"
    # Optional confidence score from the model
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ExtractItemResult(BaseModel):
    id: Optional[str]
    skills: List[SkillSpan]


class ExtractResponse(BaseModel):
    model_used: str
    results: List[ExtractItemResult]


# -----------------------------------------------------------------------------
# Model service layer (plug your real models here)
# -----------------------------------------------------------------------------
class ClassifierService:
    """
    Wraps both classical (TF-IDF + Logistic Regression) and modern (BERT) pipelines.
    Replace the dummy logic with real .predict() using your trained artifacts.
    """
    def __init__(self) -> None:
        # TODO: load your vectorizer + LR checkpoint here
        self.tfidf_lr_ready = True
        # TODO: load your fine-tuned BERT (sequence classifier) checkpoint here
        self.bert_ready = True
        logger.info("ClassifierService initialized.")

    def predict(self, texts: List[str], model: str = "bert") -> List[dict]:
        """
        Returns a list of dicts with keys: level1, level2, score
        """
        if model == "tfidf_lr" and not self.tfidf_lr_ready:
            raise RuntimeError("TF-IDF+LR model is not loaded.")
        if model == "bert" and not self.bert_ready:
            raise RuntimeError("BERT classifier is not loaded.")

        # --- DUMMY IMPLEMENTATION ---
        # Replace this with actual prediction code.
        # For example:
        # probs = self.bert_model.predict_proba(texts)
        # labels = probs.argmax(axis=1)
        # map to taxonomy level1/level2
        out: List[dict] = []
        for t in texts:
            # toy heuristic: if contains 'python' -> Software/Backend Engineer
            text_low = t.lower()
            if any(k in text_low for k in ["python", "backend", "api", "django", "fastapi"]):
                out.append({"level1": "Software", "level2": "Backend Engineer", "score": 0.91})
            elif any(k in text_low for k in ["frontend", "react", "ui", "css", "javascript"]):
                out.append({"level1": "Software", "level2": "Frontend Engineer", "score": 0.88})
            elif any(k in text_low for k in ["data scientist", "ml", "machine learning", "nlp"]):
                out.append({"level1": "Data", "level2": "ML Engineer", "score": 0.89})
            else:
                out.append({"level1": "Other", "level2": "General", "score": 0.55})
        return out


class SkillExtractorService:
    """
    Wraps CRF baseline and BERT-based NER for skill spans.
    Replace the regex/dictionary toy extractor with real token classification inference.
    """
    def __init__(self) -> None:
        # TODO: load CRF and/or BERT-TokenClassification model here
        self.crf_ready = True
        self.bert_ready = True
        # You may load a skill dictionary/gazetteer built from training data for post-processing
        self.heuristic_skills = {
            "python", "sql", "pandas", "numpy", "fastapi", "django",
            "pytorch", "tensorflow", "nlp", "bert", "docker", "git"
        }
        logger.info("SkillExtractorService initialized.")

    def _find_spans_by_dict(self, text: str) -> List[SkillSpan]:
        """
        Very simple dictionary-based extractor to illustrate span offsets.
        Replace with model-driven spans (offsets from tokenizer alignment).
        """
        spans: List[SkillSpan] = []
        lower_text = text.lower()
        for skill in self.heuristic_skills:
            start = 0
            while True:
                idx = lower_text.find(skill, start)
                if idx == -1:
                    break
                spans.append(SkillSpan(text=text[idx:idx+len(skill)], start=idx, end=idx+len(skill), label="SKILL", score=0.75))
                start = idx + len(skill)
        # De-duplicate overlapping spans by picking the longest first (greedy)
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        merged: List[SkillSpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                merged.append(s)
                last_end = s.end
        return merged

    def predict(self, texts: List[str], model: str = "bert_ner", lower=True, unique=True) -> List[List[SkillSpan]]:
        """
        Returns list of list of SkillSpan for each input text.
        """
        if model == "crf" and not self.crf_ready:
            raise RuntimeError("CRF model is not loaded.")
        if model == "bert_ner" and not self.bert_ready:
            raise RuntimeError("BERT NER model is not loaded.")

        # --- DUMMY IMPLEMENTATION ---
        # Replace with real NER inference:
        # 1) tokenize with the same tokenizer used in training
        # 2) run model -> token labels (BIO)
        # 3) convert to spans (begin/end char indices) using offset mapping
        outputs: List[List[SkillSpan]] = []
        for t in texts:
            spans = self._find_spans_by_dict(t)

            # Normalize text if requested
            if lower:
                for s in spans:
                    s.text = s.text.lower()

            if unique:
                # unique by normalized text content
                seen = set()
                uniq = []
                for s in spans:
                    key = s.text
                    if key not in seen:
                        uniq.append(s)
                        seen.add(key)
                spans = uniq

            outputs.append(spans)
        return outputs


# -----------------------------------------------------------------------------
# Application factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="TalentStream NLP API",
        version="0.1.0",
        description="FastAPI service for job classification and skill extraction."
    )

    # Services as "singletons" loaded once
    classifier = ClassifierService()
    extractor = SkillExtractorService()

    @app.get("/health")
    async def health() -> dict:
        """Simple health check for readiness/liveness probes."""
        return {"status": "ok"}

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> Any:
        try:
            texts = [it.text for it in req.items]
            preds = classifier.predict(texts, model=req.model)
            results = []
            for item, pred in zip(req.items, preds):
                results.append(ClassifyItemResult(
                    id=item.id,
                    level1=pred["level1"],
                    level2=pred["level2"],
                    score=pred.get("score")
                ))
            return ClassifyResponse(model_used=req.model, results=results)
        except RuntimeError as e:
            logger.exception("Model runtime error")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("Unhandled error in /classify")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.post("/extract_skills", response_model=ExtractResponse)
    async def extract_skills(req: ExtractRequest) -> Any:
        try:
            texts = [it.text for it in req.items]
            batch_spans = extractor.predict(
                texts, model=req.model, lower=req.lower, unique=req.unique
            )
            results = []
            for item, spans in zip(req.items, batch_spans):
                results.append(ExtractItemResult(id=item.id, skills=spans))
            return ExtractResponse(model_used=req.model, results=results)
        except RuntimeError as e:
            logger.exception("Model runtime error")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("Unhandled error in /extract_skills")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    return app


app = create_app()

# If you prefer "python app.py" to run a local dev server:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)

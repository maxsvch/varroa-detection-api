from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    confidence: float


class AnalysisResponse(BaseModel):
    varroa_count: int
    confidence_mean: float = Field(ge=0.0, le=1.0)
    detections: list[DetectionBox]
    processing_time_ms: int

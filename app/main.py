import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.detector import analyze_image, load_model
from app.schemas import AnalysisResponse

MODEL_PATH = os.getenv("MODEL_PATH", "model/weights/best.pt")
VARROA_API_KEY = os.getenv("VARROA_API_KEY", "")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Varroa Detection API",
    description=(
        "REST API wrapping the VarroDetector YOLOv11 nano model for Varroa mite "
        "detection in beehive sticky board images. "
        "Source: https://github.com/jodivaso/varrodetector — Licence AGPL-3.0."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _check_api_key(x_api_key: str | None) -> None:
    if VARROA_API_KEY and x_api_key != VARROA_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    content = await image.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    filename = image.filename or "image.jpg"
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".jpg"

    try:
        result = analyze_image(content, suffix, CONFIDENCE_THRESHOLD)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return JSONResponse(content=result)

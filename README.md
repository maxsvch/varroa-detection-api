# varroa-detection-api

REST API for automatic Varroa mite detection in beehive sticky board images.

Built on top of [VarroDetector](https://github.com/jodivaso/varrodetector) by J. Díaz-Vaso et al., wrapping the YOLOv11 nano model in a FastAPI service.

**Licence : AGPL-3.0** — see [LICENSE](./LICENSE).

---

## What it does

Send a photo of a sticky board (plancher de ruche) and get back the count of detected Varroa mites with bounding box coordinates and confidence scores.

```
POST /analyze
Content-Type: multipart/form-data
X-API-Key: <secret>

image: <file>  # JPG, PNG, HEIC — max 10 MB

→ {
    "varroa_count": 47,
    "confidence_mean": 0.82,
    "detections": [
      { "x": 120.0, "y": 340.0, "w": 18.0, "h": 14.0, "confidence": 0.91 },
      ...
    ],
    "processing_time_ms": 1240
  }
```

---

## Model weights

The model weights (`best.pt`, 9.15 MB) are not versioned in this repository. Download them from the [VarroDetector releases](https://github.com/jodivaso/varrodetector/releases) and place the file at:

```
model/weights/best.pt
```

---

## Run locally

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download model weights → model/weights/best.pt

# 3. Configure
cp .env.example .env
# Edit VARROA_API_KEY in .env

# 4. Start
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## Docker

```bash
docker build -t varroa-detection-api .
docker run -p 8000:8000 \
  -e VARROA_API_KEY=secret \
  -e CONFIDENCE_THRESHOLD=0.1 \
  -v $(pwd)/model:/app/model \
  varroa-detection-api
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VARROA_API_KEY` | `` | Required secret key (header `X-API-Key`) |
| `CONFIDENCE_THRESHOLD` | `0.1` | YOLO detection confidence threshold (0.0–1.0) |
| `MODEL_PATH` | `model/weights/best.pt` | Path to YOLOv11 weights |

---

## Credits

- **VarroDetector** — J. Díaz-Vaso et al. — [github.com/jodivaso/varrodetector](https://github.com/jodivaso/varrodetector) — AGPL-3.0
- **YOLOv11** — Ultralytics — Apache 2.0 / AGPL-3.0

---

## Licence

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

If you run a modified version of this service over a network, you must make your modified source code available to users of that service. See [LICENSE](./LICENSE) for full terms.

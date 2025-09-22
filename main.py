from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import tempfile
import cv2
from datetime import datetime, timedelta
from typing import Dict
import asyncio
import logging

# Import model logic
from predict import classify_objects, estimate_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Waste Detection API",
    description="AI-powered waste classification and weight estimation",
    version="1.0.0"
)

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Template setup
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory storage for temporary images
TEMP_IMAGES: Dict[str, bytes] = {}
TEMP_IMAGES_EXPIRY: Dict[str, datetime] = {}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Waste Detection API started successfully!")
    asyncio.create_task(cleanup_expired_images())

@app.on_event("shutdown")
async def shutdown_event():
    TEMP_IMAGES.clear()
    TEMP_IMAGES_EXPIRY.clear()
    for filename in os.listdir("uploads"):
        file_path = os.path.join("uploads", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
    logger.info("🧹 Shutdown: Cleaned temporary data")

async def cleanup_expired_images():
    while True:
        try:
            await asyncio.sleep(300)
            now = datetime.now()
            expired_ids = [img_id for img_id, expiry in TEMP_IMAGES_EXPIRY.items() if now > expiry]
            for img_id in expired_ids:
                TEMP_IMAGES.pop(img_id, None)
                TEMP_IMAGES_EXPIRY.pop(img_id, None)
            if expired_ids:
                logger.info(f"🧹 Cleaned {len(expired_ids)} expired annotated images")
        except Exception as e:
            logger.error(f"⚠️ Background cleanup failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file_web(request: Request, file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise ValueError("No file selected")

        ext = file.filename.split('.')[-1].lower() if '.' in file.filename else 'jpg'
        if ext not in {"jpg", "jpeg", "png", "bmp"}:
            raise ValueError("Unsupported file type. Use JPG, PNG, or BMP.")

        unique_filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join("uploads", unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = classify_objects(file_path)

        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError(f"OpenCV could not load image: {file_path}")

        h, w = img_cv.shape[:2]

        color_map = {
            "Dry Waste": (0, 255, 0),
            "Wet Waste": (0, 255, 255),
            "Hazardous Waste": (0, 0, 255),
            "Electronic Waste": (255, 0, 255),
            "Construction Waste": (255, 255, 0),
            "Biomedical Waste": (128, 0, 128)
        }

        for obj in results:
            category = obj["category"]
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = color_map.get(category, (0, 255, 0))

            # Calculate 50x50 box centered at (cx, cy)
            x1_box = max(0, min(cx - 25, w - 50))
            y1_box = max(0, min(cy - 25, h - 50))
            x2_box = x1_box + 50
            y2_box = y1_box + 50

            # # Draw bounding box
            # cv2.rectangle(img_cv, (x1_box, y1_box), (x2_box, y2_box), color, 2)

            # Draw center dot
            cv2.circle(img_cv, (cx, cy), 5, color, -1)

            # Draw category label near center point
            label = f"{category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
            label_x = max(5, min(cx - text_w // 2, w - text_w - 5))
            label_y = max(20, min(cy - 30, h - 5))

            overlay = img_cv.copy()
            cv2.rectangle(overlay, (label_x - 5, label_y - text_h - 5),
                         (label_x + text_w + 5, label_y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)
            cv2.putText(img_cv, label, (label_x, label_y), font, 0.6, (255, 255, 255), 2)

        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise RuntimeError("Failed to encode annotated image")

        temp_id = str(uuid.uuid4())
        TEMP_IMAGES[temp_id] = buffer.tobytes()
        TEMP_IMAGES_EXPIRY[temp_id] = datetime.now() + timedelta(minutes=30)

        px_to_cm = 0.1
        formatted_results = []
        for obj in results:
            bbox = obj['bbox']
            area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_cm2 = area_px * (px_to_cm ** 2)
            weight_info = ""
            if obj['category'] not in ["Hazardous Waste", "Construction Waste"]:
                weight_kg = estimate_weight(obj['category'], area_cm2)
                weight_info = f" ⚖️ Tentative Weight: {weight_kg:.2f} kg"

            formatted_results.append({
                'object': obj['object'],
                'category': obj['category'],
                'area': f"{area_cm2:.1f}",
                'weight_info': weight_info,
                'disposal': obj['disposal'],
                'color': {
                    "Dry Waste": "#00FF00",
                    "Wet Waste": "#FFFF00",
                    "Hazardous Waste": "#FF0000",
                    "Electronic Waste": "#FF00FF",
                    "Construction Waste": "#00FFFF",
                    "Biomedical Waste": "#800080"
                }.get(obj['category'], "#FFFFFF")
            })

        os.unlink(file_path)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image": unique_filename,
            "annotated_image_id": temp_id,
            "results": formatted_results
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        })

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse({"success": False, "error": "No file provided"}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name

        results = classify_objects(temp_path)

        px_to_cm = 0.1
        for obj in results:
            bbox = obj['bbox']
            area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_cm2 = area_px * (px_to_cm ** 2)
            obj['area_cm2'] = round(area_cm2, 1)
            if obj['category'] not in ["Hazardous Waste", "Construction Waste"]:
                obj['tentative_weight_kg'] = round(estimate_weight(obj['category'], area_cm2), 2)

        os.unlink(temp_path)
        return {"success": True, "count": len(results), "results": results}

    except Exception as e:
        logger.error(f"Prediction API failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/temp_image/{image_id}")
async def get_temp_image(image_id: str):
    if image_id not in TEMP_IMAGES:
        return JSONResponse({"error": "Image not found"}, status_code=404)

    if datetime.now() > TEMP_IMAGES_EXPIRY[image_id]:
        del TEMP_IMAGES[image_id]
        del TEMP_IMAGES_EXPIRY[image_id]
        return JSONResponse({"error": "Image expired"}, status_code=410)

    return Response(content=TEMP_IMAGES[image_id], media_type="image/jpeg")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Waste Detection API running", "version": "1.0.0"}

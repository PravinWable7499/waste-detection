# main.py
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
import base64
from datetime import datetime, timedelta
from typing import Dict
import asyncio
import logging

# Import your model logic
from predict import classify_objects, estimate_weight

# ======================
# Setup Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# App Initialization
# ======================
app = FastAPI(
    title="Waste Detection API",
    description="AI-powered waste classification and weight estimation",
    version="1.0.0"
)

# ======================
# Middleware
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 Lock down in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Static Files & Templates
# ======================
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# In-Memory Storage (for demo)
# ======================
TEMP_IMAGES: Dict[str, bytes] = {}
TEMP_IMAGES_EXPIRY: Dict[str, datetime] = {}

# ======================
# Startup & Cleanup
# ======================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Waste Detection API started successfully!")
    asyncio.create_task(cleanup_expired_images())

@app.on_event("shutdown")
async def shutdown_event():
    TEMP_IMAGES.clear()
    TEMP_IMAGES_EXPIRY.clear()
    # Clean uploads folder
    for filename in os.listdir("uploads"):
        file_path = os.path.join("uploads", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
    logger.info("🧹 Shutdown: Cleaned temporary data")

async def cleanup_expired_images():
    """Background task to clean expired images every 5 minutes."""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            now = datetime.now()
            expired_ids = [
                img_id for img_id, expiry in TEMP_IMAGES_EXPIRY.items()
                if now > expiry
            ]
            for img_id in expired_ids:
                TEMP_IMAGES.pop(img_id, None)
                TEMP_IMAGES_EXPIRY.pop(img_id, None)
            if expired_ids:
                logger.info(f"🧹 Cleaned {len(expired_ids)} expired annotated images")
        except Exception as e:
            logger.error(f"⚠️ Background cleanup failed: {e}")


# ======================
# Routes
# ======================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file_web(request: Request, file: UploadFile = File(...)):
    """Handle image upload via web form, run detection, and render results."""
    try:
        # Validate file
        if not file.filename:
            raise ValueError("No file selected")

        # Generate safe filename
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'jpg'
        if file_extension not in {'jpg', 'jpeg', 'png', 'bmp'}:
            raise ValueError("Unsupported file type. Use JPG, PNG, or BMP.")

        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join("uploads", unique_filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model
        try:
            results = classify_objects(file_path)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")

        # Load image with OpenCV
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError(f"OpenCV could not load image: {file_path}")

        # Annotate image
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
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            color = color_map.get(category, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"{category}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.7, 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            label_y = y1 - 10 if y1 > text_h + 10 else y2 + 20
            label_x = min(x1, img_cv.shape[1] - text_w - 5)

            overlay = img_cv.copy()
            cv2.rectangle(overlay,
                         (label_x - 5, label_y - text_h - 5),
                         (label_x + text_w + 5, label_y + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)
            cv2.putText(img_cv, label, (label_x, label_y),
                       font, font_scale, (255, 255, 255), thickness)

            # Draw center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img_cv, (cx, cy), 4, color, -1)

        # Encode image to JPEG
        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise RuntimeError("Failed to encode annotated image")

        # Store in memory
        temp_id = str(uuid.uuid4())
        TEMP_IMAGES[temp_id] = buffer.tobytes()
        TEMP_IMAGES_EXPIRY[temp_id] = datetime.now() + timedelta(minutes=30)

        # Format results for template
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

        # Cleanup original upload
        os.unlink(file_path)

        # Render result page
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
    """API endpoint for raw JSON prediction (for mobile/apps)."""
    try:
        if not file.filename:
            return JSONResponse({"success": False, "error": "No file provided"}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name

        try:
            results = classify_objects(temp_path)
        except Exception as e:
            return JSONResponse({"success": False, "error": f"Model error: {str(e)}"}, status_code=500)

        # Add area & weight
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
    """Serve annotated image from memory."""
    if image_id not in TEMP_IMAGES:
        return JSONResponse({"error": "Image not found"}, status_code=404)

    if datetime.now() > TEMP_IMAGES_EXPIRY[image_id]:
        del TEMP_IMAGES[image_id]
        del TEMP_IMAGES_EXPIRY[image_id]
        return JSONResponse({"error": "Image expired"}, status_code=410)

    return Response(content=TEMP_IMAGES[image_id], media_type="image/jpeg")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "message": "Waste Detection API running",
        "version": "1.0.0",
        "uptime": "since startup"
    }
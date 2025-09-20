# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import tempfile
import cv2
import base64
import socket
from datetime import datetime, timedelta
from typing import Dict
from predict import classify_objects, estimate_weight

app = FastAPI(title="Waste Detection API")

#  Allow cross-origin for dev (use specific domain/IP for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs("uploads", exist_ok=True)

# In-memory storage for annotated images
TEMP_IMAGES: Dict[str, bytes] = {}
TEMP_IMAGES_EXPIRY: Dict[str, datetime] = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file_web(request: Request, file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"uploads/{unique_filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = classify_objects(file_path)
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise Exception("Failed to load image")

        for obj in results:
            category = obj["category"]
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            color = {
                "Dry Waste": (0, 255, 0),
                "Wet Waste": (0, 255, 255),
                "Hazardous Waste": (0, 0, 255),
                "Electronic Waste": (255, 0, 255),
                "Construction Waste": (255, 255, 0),
                "Biomedical Waste": (128, 0, 128)
            }.get(category, (0, 255, 0))

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
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

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img_cv, (cx, cy), 4, color, -1)

        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise Exception("Failed to encode image")

        temp_id = str(uuid.uuid4())
        TEMP_IMAGES[temp_id] = buffer.tobytes()
        TEMP_IMAGES_EXPIRY[temp_id] = datetime.now() + timedelta(minutes=30)

        formatted_results = []
        px_to_cm = 0.1
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
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    try:
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
        return {"success": False, "error": str(e)}

@app.get("/temp_image/{image_id}")
async def get_temp_image(image_id: str):
    if image_id not in TEMP_IMAGES:
        return {"error": "Image not found or expired"}
    if datetime.now() > TEMP_IMAGES_EXPIRY[image_id]:
        del TEMP_IMAGES[image_id]
        del TEMP_IMAGES_EXPIRY[image_id]
        return {"error": "Image expired"}
    return Response(content=TEMP_IMAGES[image_id], media_type="image/jpeg")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Waste Detection API running", "version": "1.0"}

import asyncio
@app.on_event("startup")
async def cleanup_temp_images():
    async def cleanup():
        while True:
            await asyncio.sleep(3600)
            now = datetime.now()
            expired_ids = [k for k, exp in TEMP_IMAGES_EXPIRY.items() if now > exp]
            for k in expired_ids:
                TEMP_IMAGES.pop(k, None)
                TEMP_IMAGES_EXPIRY.pop(k, None)
            print(f"🧹 Cleaned {len(expired_ids)} expired images")
    asyncio.create_task(cleanup())

if __name__ == "__main__":
    import uvicorn
    ip = socket.gethostbyname(socket.gethostname())  # Get local IP
    print(f"🌐 Access API on: http://{ip}:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

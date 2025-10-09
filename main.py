from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
import uuid
import tempfile
import cv2
from datetime import datetime, timedelta
from typing import Dict
import asyncio
import logging
from predict import translate_to_marathi  # Import at top of main.py
from predict import classify_objects, estimate_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Waste Detection API",
    description="AI-powered waste classification and weight estimation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Near the top of main.py, after imports
# OBJECT_NAME_TRANSLATIONS_MR = {
#     # Wet Waste
#     "banana peel": "केळ्याचे साल",
#     "apple peel": "सफरचंदाचे साल",
#     "lime peel": "लिंबाचे साल",
#     "lemon peel": "लिंबाचे साल",
#     "orange peel": "संत्र्याचे साल",
#     "mango peel": "आंब्याचे साल",
#     "potato peel": "बटाट्याचे साल",
#     "onion peel": "कांद्याचे साल",
#     "garlic peel": "लसूणाचे साल",
#     "vegetable scraps": "भाजीचे उरे",
#     "fruit scraps": "फळांचे उरे",
#     "leftover food": "उरलेले अन्न",
#     "tea leaves": "चहाची पाने",
#     "coffee grounds": "कॉफीचा गर",
#     "egg shell": "अंड्याचे कवच",
#     "lime pulp": "चूनाचा गर",

#     # Dry Waste
#     "plastic bottle": "प्लास्टिकची बाटली",
#     "glass bottle": "काचेची बाटली",
#     "aluminum can": "अॅल्युमिनियमची कॅन",
#     "paper cup": "कागदी कप",
#     "cardboard box": "कार्डबोर्डचा डबा",
#     "newspaper": "वृत्तपत्र",
#     "magazine": "मासिक",
#     "plastic bag": "प्लास्टिकची पिशवी",

#     # E-Waste
#     "old phone": "जुना मोबाईल",
#     "broken charger": "बिघाडलेला चार्जर",
#     "dead battery": "संपलेली बॅटरी",
#     "old laptop": "जुने लॅपटॉप",

    # Add more as needed...
# }

# === Category names for localization ===
CATEGORY_NAMES = {
    "en": {
        "Dry Waste": "Dry Waste",
        "Wet Waste": "Wet Waste",
        "Hazardous Waste": "Hazardous Waste",
        "Electronic Waste": "Electronic Waste",
        "Construction Waste": "Construction Waste",
        "Biomedical Waste": "Biomedical Waste"
    },
    "mr": {
        "Dry Waste": "सुका कचरा",
        "Wet Waste": "ओला कचरा",
        "Hazardous Waste": "धोकादायक कचरा",
        "Electronic Waste": "इ-कचरा",
        "Construction Waste": "बांधकाम कचरा",
        "Biomedical Waste": "जैविक कचरा"
    }
}

json_file_path = "data.json"
jsonDb = {}

if os.path.exists(json_file_path):
    try:
        with open(json_file_path, "r",encoding="utf-8") as f:
            jsonDb = json.load(f)
    except json.JSONDecodeError:
        jsonDb = {}

os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TEMP_IMAGES: Dict[str, bytes] = {}
TEMP_IMAGES_EXPIRY: Dict[str, datetime] = {}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Waste Detection API started!")
    asyncio.create_task(cleanup_expired_images())

@app.on_event("shutdown")
async def shutdown_event():
    TEMP_IMAGES.clear()
    TEMP_IMAGES_EXPIRY.clear()
    for filename in os.listdir("uploads"):
        try:
            os.unlink(os.path.join("uploads", filename))
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
    logger.info("🧹 Shutdown: Cleaned temporary data")

async def cleanup_expired_images():
    while True:
        await asyncio.sleep(300)
        now = datetime.now()
        expired = [k for k, v in TEMP_IMAGES_EXPIRY.items() if now > v]
        for k in expired:
            TEMP_IMAGES.pop(k, None)
            TEMP_IMAGES_EXPIRY.pop(k, None)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/getHistory")
def get_all_data():
    return jsonDb

@app.post("/upload")
async def upload_file_web(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Form("en"),
    isJson: str = Form("false")
):
    
    try:
        is_json = isJson.lower() == "true"

        if not file.filename:
            raise ValueError("No file selected")

        ext = file.filename.split('.')[-1].lower() if '.' in file.filename else 'jpg'
        if ext not in {"jpg", "jpeg", "png", "bmp"}:
            raise ValueError("Unsupported file type")

        unique_filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join("uploads", unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = classify_objects(file_path, lang=lang)

        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError("OpenCV could not load image")

        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise RuntimeError("Failed to encode image")

        temp_id = str(uuid.uuid4())
        TEMP_IMAGES[temp_id] = buffer.tobytes()
        TEMP_IMAGES_EXPIRY[temp_id] = datetime.now() + timedelta(minutes=30)

        formatted_results = []
        for obj in results:
            area_cm2 = obj.get('area_cm2', 0.0)
            weight_kg = obj.get('tentative_weight_kg', 0.0) or 0.0

            object_name = obj['object']
            if obj.get('count', 1) > 1:
                object_name = f"Set of {object_name}s"

            original_category = obj['category']
            localized_category = CATEGORY_NAMES.get(lang, CATEGORY_NAMES["en"]).get(original_category, original_category)

            entry = {
                'object': object_name,
                'category': localized_category,
                'area_cm2': round(area_cm2, 1),
                'weight_kg': round(weight_kg, 2),
                'disposal': obj['disposal'],
                'color': {
                    "Dry Waste": "#00FF00",
                    "Wet Waste": "#FFFF00",
                    "Hazardous Waste": "#FF0000",
                    "Electronic Waste": "#FF00FF",
                    "Construction Waste": "#00FFFF",
                    "Biomedical Waste": "#800080"
                }.get(original_category, "#FFFFFF")
            }
            formatted_results.append(entry)

            if lang == 'mr':
                marathi_object = translate_to_marathi(obj['object'])
                history_entry = {
                    'वस्तू': marathi_object,
                    'श्रेणी': localized_category,
                    'क्षेत्रफळ': f"{area_cm2:.1f} सेमी²",
                    'अंदाजे वजन': f"{weight_kg:.2f} किलोग्राम",
                    'विल्हेवाट': obj['disposal'],
                    'रंग': entry['color']
                }
            else:
                history_entry = {
                    'object': obj['object'],
                    'category': localized_category,
                    'Area': f"{area_cm2:.1f} cm²",
                    'Tentative Weight': f"{weight_kg:.2f} kg",
                    'disposal': obj['disposal'],
                    'color': entry['color']
                }

            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            jsonDb[f"{current_datetime}_{len(jsonDb)}"] = [history_entry]

        with open(json_file_path, "w") as f:
            json.dump(jsonDb, f, indent=4)

        # === JSON RESPONSE IF REQUESTED ==   
        accept_header = request.headers.get("accept", "").lower()
        if is_json or "application/json" in accept_header:
           
            return JSONResponse({
                "original_image": unique_filename,
                "annotated_image_id": temp_id,
                "results": formatted_results,
                "lang": lang
            })

        # === HTML TEMPLATE RESPONSE ===
        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image": unique_filename,
            "annotated_image_id": temp_id,
            "results": results,
            "lang": lang
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if is_json:
            return JSONResponse({"error": str(e)}, status_code=400)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...), lang: str = "en"):
    try:
        if not file.filename:
            return JSONResponse({"success": False, "error": "No file"}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        results = classify_objects(temp_path, lang=lang)

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
        logger.error(f"Prediction failed: {e}")
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
    return {"status": "healthy", "version": "1.0.0"}
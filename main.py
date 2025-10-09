from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json, os, shutil, uuid, tempfile, cv2, asyncio, logging
from datetime import datetime, timedelta
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

# === Import core logic ===
from predict import classify_objects, estimate_weight

# === Setup ===
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

# === Localization ===
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

# === Global state ===
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TEMP_IMAGES: Dict[str, bytes] = {}
TEMP_IMAGES_EXPIRY: Dict[str, datetime] = {}

# Use threadpool for heavy tasks (Gemini calls, OpenCV)
executor = ThreadPoolExecutor(max_workers=4)


# === Background Cleanup ===
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


# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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

        # Fast file write using async thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, lambda: shutil.copyfileobj(file.file, open(file_path, "wb")))

        # Run classification asynchronously with retry
        results = await run_classification_with_retry(file_path, lang)

        # Handle OpenCV encoding in threadpool
        img_cv = await loop.run_in_executor(executor, lambda: cv2.imread(file_path))
        if img_cv is None:
            raise ValueError("OpenCV could not load image")

        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise RuntimeError("Failed to encode image")

        temp_id = str(uuid.uuid4())
        TEMP_IMAGES[temp_id] = buffer.tobytes()
        TEMP_IMAGES_EXPIRY[temp_id] = datetime.now() + timedelta(minutes=30)

        # Format results
        formatted_results = []
        for obj in results or []:
            area_cm2 = obj.get('area_cm2', 0.0)
            weight_kg = obj.get('tentative_weight_kg', 0.0) or 0.0
            object_name = obj['object']
            if obj.get('count', 1) > 1:
                object_name = f"Set of {object_name}s"

            original_category = obj['category']
            localized_category = CATEGORY_NAMES.get(lang, CATEGORY_NAMES["en"]).get(original_category, original_category)
            formatted_results.append({
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
            })

        # === JSON RESPONSE IF REQUESTED ===
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

        # Run classifier in background thread
        results = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: classify_objects(temp_path, lang=lang)
        )

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


# === Helper: Retry Gemini classification ===
async def run_classification_with_retry(file_path: str, lang: str, retries: int = 2, timeout: int = 60):
    """
    Run classify_objects() in a thread with retry and timeout.
    Prevents null/empty outputs from Gemini due to timeouts or API stalls.
    """
    loop = asyncio.get_event_loop()
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: classify_objects(file_path, lang=lang)),
                timeout=timeout
            )
            if result:  # non-empty result
                return result
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout on attempt {attempt + 1} for {file_path}")
        except Exception as e:
            logger.error(f"Classification failed on attempt {attempt + 1}: {e}")

        await asyncio.sleep(2)  # small backoff

    logger.error("❌ All retries failed — returning empty result")
    return []

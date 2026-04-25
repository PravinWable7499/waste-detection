import os
import cv2
import json
import re
from PIL import Image
from collections import defaultdict
import google.generativeai as genai

# ========================
# 🔹 CONFIGURATION
# ========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

GOOGLE_API_KEY = "AIzaSyDTjKDxjZK02raiHrzYbAlGu1n1lM-ptag"
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_ID = "gemini-2.5-flash-lite"

# 🔥 LOAD MODEL ONLY ONCE (BIG SPEED BOOST)
model = genai.GenerativeModel(MODEL_ID)

# ========================
# 🔹 CONSTANTS
# ========================
WASTE_CATEGORIES = {
    "Dry Waste", "Wet Waste", "Hazardous Waste",
    "Electronic Waste", "Construction Waste", "Biomedical Waste"
}

COLOR_MAP = {
    "Dry Waste": (0, 255, 0),
    "Wet Waste": (0, 255, 255),
    "Hazardous Waste": (0, 0, 255),
    "Electronic Waste": (255, 0, 255),
    "Construction Waste": (255, 255, 0),
    "Biomedical Waste": (128, 0, 128),
}

# ========================
# 🔹 DISPOSAL GUIDE
# ========================
DISPOSAL_GUIDE = {
    "en": {
        "Dry Waste": ["Recycle items", "Compost if possible", "Use dry bin"],
        "Wet Waste": ["Compost", "Use kitchen bin", "Give to collection"],
        "Hazardous Waste": ["Use collection center", "Do not mix", "Seal waste"],
        "Electronic Waste": ["Recycle properly", "Do not throw", "Use e-waste centers"],
        "Construction Waste": ["Use removal service", "Do not dump", "Reuse material"],
        "Biomedical Waste": ["Use authorized disposal", "Avoid mixing", "Use yellow bins"]
    }
}

# ========================
# 🔹 PROMPT
# ========================
DETECTION_PROMPT = """
Return JSON:
[{"object":"","category":"","bbox":[x1,y1,x2,y2]}]

Rules:
- Group similar items
- Integer bbox
- Correct category
- JSON only
"""

# ========================
# 🔹 HELPERS (OPTIMIZED)
# ========================

def get_image_dimensions(path):
    img = cv2.imread(path)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def estimate_weight(category, area):
    density = {
        "Dry Waste": 0.001,
        "Wet Waste": 0.001,
        "Hazardous Waste": 0.002,
        "Electronic Waste": 0.003,
        "Construction Waste": 0.004,
        "Biomedical Waste": 0.0015
    }.get(category, 0.001)

    weight = area * 10 * density
    return round(min(weight, 5), 2)


def aggregate_objects(data, lang="en"):
    grouped = defaultdict(list)

    for obj in data:
        grouped[(obj["object"], obj["category"])].append(obj)

    result = []
    for (name, cat), items in grouped.items():
        area = sum(i["area_cm2"] for i in items)
        weight = sum(i["tentative_weight_kg"] for i in items)

        result.append({
            "object": name,
            "category": cat,
            "area_cm2": round(area, 1),
            "tentative_weight_kg": round(weight, 2),
            "count": len(items),
            "color": "#{:02X}{:02X}{:02X}".format(*COLOR_MAP[cat]),
            "disposal": DISPOSAL_GUIDE["en"].get(cat, [])
        })

    return result


# ========================
# 🔹 MAIN FUNCTION (FAST)
# ========================

def classify_objects(image_path, lang="en"):
    try:
        w, h = get_image_dimensions(image_path)
        if not w:
            return []

        img = Image.open(image_path)

        # 🔥 FAST API CALL
        response = model.generate_content([DETECTION_PROMPT, img])

        text = response.text.strip()

        # 🔥 FAST CLEAN (no regex heavy)
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()

        try:
            detections = json.loads(text)
        except:
            return []

        validated = []

        for obj in detections:
            cat = obj.get("category")
            bbox = obj.get("bbox")

            if cat not in WASTE_CATEGORIES or not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            area_px = max(0, x2 - x1) * max(0, y2 - y1)
            area_cm2 = area_px * 0.0007  # precomputed factor

            validated.append({
                "object": obj.get("object", "Unknown"),
                "category": cat,
                "bbox": [x1, y1, x2, y2],
                "area_cm2": area_cm2,
                "tentative_weight_kg": estimate_weight(cat, area_cm2)
            })

        return aggregate_objects(validated, lang)

    except Exception as e:
        print("Error:", e)
        return []


# ========================
# 🔹 DRAW FUNCTION (UNCHANGED)
# ========================

def draw_annotations(image_path, results, output_path="annotated.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = COLOR_MAP[r["category"]]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, r["category"], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(output_path, img)
    return output_path

import os
import cv2
from PIL import Image
import google.generativeai as genai
import json
import re
from collections import defaultdict  # <-- ADDED

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GOOGLE_API_KEY = 'AIzaSyDTjKDxjZK02raiHrzYbAlGu1n1lM-ptag'
genai.configure(api_key=GOOGLE_API_KEY)

WASTE_CATEGORIES = [
    "Dry Waste",
    "Wet Waste",
    "Hazardous Waste",
    "Electronic Waste",
    "Construction Waste",
    "Biomedical Waste"
]

COLOR_MAP = {
    "Dry Waste": (0, 255, 0),
    "Wet Waste": (0, 255, 255),
    "Hazardous Waste": (0, 0, 255),
    "Electronic Waste": (255, 0, 255),
    "Construction Waste": (255, 255, 0),
    "Biomedical Waste": (128, 0, 128)
}

DISPOSAL_GUIDE = {
    "Dry Waste": "📦 Recycle or compost if biodegradable. Place in dry waste bin.",
    "Wet Waste": "🍃 Compost at home or in municipal composting facility.",
    "Hazardous Waste": "⚠️ Take to designated hazardous waste collection center. Do not throw in regular trash.",
    "Electronic Waste": "🔋 Return to e-waste recycling center. Never dispose in landfill.",
    "Construction Waste": "🏗️ Hire a licensed waste removal service. Do not dump illegally.",
    "Biomedical Waste": "🩺 Dispose only through medical waste services. Never discard in household bins."
}

DETECTION_PROMPT = """
You are a professional waste classification expert specializing in environmental sustainability.

 IMPORTANT INSTRUCTIONS:
- Analyze the image carefully.
- Identify ALL visible waste items with PRECISE bounding boxes.
- Each object must be classified accurately based on its physical properties.
- Every object must have its own tight bounding box around only that object.
- Bounding boxes should present center of detetcted object

 Object Classification Rules:
- Wet Waste: Food scraps, peels, rotten fruits, organic matter
- Dry Waste: Paper, cardboard, plastic bottles, metal cans, glass, packaging
- Hazardous Waste: Batteries, chemicals, paint, medicine, syringes
- Electronic Waste: Phones, wires, circuits, chargers, old electronics
- Construction Waste: Concrete, bricks, tiles, wood, rubble
- Biomedical Waste: Syringes, medical containers, expired medicines

 Coordinate System:
- Top-left corner is (0,0)
- Format: [x_min, y_min, x_max, y_max]
- x_min < x_max, y_min < y_max
- Coordinates must tightly fit the object — no extra padding

 Output Format:
Return ONLY a valid JSON array with these fields:
[
  {
    "object": "rotten banana peel",
    "category": "Wet Waste",
    "bbox": [100, 50, 300, 200],
    "disposal": "Compost at home or in municipal composting facility."
  }
]

It should be precise bounding boxes because sometimes it doesn’t work

No explanations, markdown, or extra text.
"""

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height

# ✅ ADDED: Aggregation function
def aggregate_objects(results):
    grouped = defaultdict(list)
    for obj in results:
        key = (obj["object"], obj["category"])
        grouped[key].append(obj)
    
    aggregated = []
    for (name, category), items in grouped.items():
        total_area = sum(obj.get("area_cm2", 0) for obj in items)
        total_weight = sum(obj.get("tentative_weight_kg", 0) for obj in items)
        count = len(items)
        # Use first item's disposal and bbox
        first = items[0]
        aggregated.append({
            "object": name,
            "category": category,
            "bbox": first["bbox"],
            "disposal": first["disposal"],
            "area_cm2": round(total_area, 1),
            "tentative_weight_kg": round(total_weight, 2),
            "count": count
        })
    return aggregated

def classify_objects(image_path):
    try:
        img_width, img_height = get_image_dimensions(image_path)
        if img_width is None:
            return []

        img_pil = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        dimension_prompt = f"{DETECTION_PROMPT}\n\nImage dimensions: {img_width} × {img_height} pixels"

        response = model.generate_content([dimension_prompt, img_pil])
        text = response.text.strip()

        cleaned_text = re.sub(r'^```(?:json|py|javascript)?\s*\n', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\n\s*```\s*$', '', cleaned_text).strip()

        results = []
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                results = parsed
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            return []

        validated_results = []
        for obj in results:
            if not isinstance(obj, dict):
                continue
            category = obj.get("category", "").strip()
            bbox = obj.get("bbox", [])
            obj_name = obj.get("object", "Unknown")
            disposal = obj.get("disposal", "")

            if category not in WASTE_CATEGORIES:
                print(f"Invalid category: '{category}' — skipping.")
                continue

            if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                print(f"Invalid bbox: {bbox} — skipping.")
                continue

            x1, y1, x2, y2 = map(int, bbox)
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bbox dimensions: {bbox} — skipping.")
                continue

            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))

            if not disposal:
                disposal = DISPOSAL_GUIDE.get(category, "Dispose responsibly in appropriate bin.")

            validated_results.append({
                "object": obj_name,
                "category": category,
                "bbox": [x1, y1, x2, y2],
                "disposal": disposal
            })

        # ✅ AGGREGATE HERE
        validated_results = aggregate_objects(validated_results)
        return validated_results

    except Exception as e:
        print(f"Error during classification: {e}")
        return []

def estimate_weight(category, area_cm2):
    densities = {
        "Dry Waste": 0.001,
        "Wet Waste": 0.001,
        "Hazardous Waste": 0.002,
        "Electronic Waste": 0.003,
        "Construction Waste": 0.004,
        "Biomedical Waste": 0.0015
    }

    depth_cm = 10
    volume_cm3 = area_cm2 * depth_cm
    density = densities.get(category, 0.001)
    weight_kg = density * volume_cm3

    max_weight = {
        "Dry Waste": 1.0,
        "Wet Waste": 0.8,
        "Hazardous Waste": 2.0,
        "Electronic Waste": 1.5,
        "Construction Waste": 5.0,
        "Biomedical Waste": 0.5
    }.get(category, 1.0)

    return min(weight_kg, max_weight)

def draw_annotations(image_path, results, output_path="annotated_result.jpg"):
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return None

    height, width = img_cv.shape[:2]

    for obj in results:
        category = obj["category"]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = bbox

        color = COLOR_MAP[category]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img_cv, (cx, cy), 6, color, -1)

        label = f"{category}"
        font_scale = 0.6
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = cy - 15
        label_x = cx - text_w // 2

        label_x = max(5, min(label_x, width - text_w - 5))
        label_y = max(text_h + 5, min(label_y, height - 10))

        overlay = img_cv.copy()
        cv2.rectangle(overlay,
                     (label_x - 5, label_y - text_h - 5),
                     (label_x + text_w + 5, label_y + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)

        cv2.putText(img_cv, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness)

    cv2.imwrite(output_path, img_cv)
    return output_path
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

MODEL_ID = "gemini-2.5-flash-lite"   # ⚡ Faster & more stable than 2.5-lite


# ========================
# 🔹 WASTE CATEGORY DEFINITIONS
# ========================
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
    "Biomedical Waste": (128, 0, 128),
}

# ========================
# 🔹 DISPOSAL INSTRUCTIONS
# ========================
DISPOSAL_GUIDE = {
    "en": {
        "Dry Waste": [
            "Recycle paper, plastic, and metal items at a recycling center.",
            "Compost biodegradable dry waste at home if suitable.",
            "Place non-recyclable dry waste in a clearly marked dry bin."
        ],
        "Wet Waste": [
            "Compost at home or in a municipal composting facility.",
            "Use a kitchen compost bin for food scraps and peels.",
            "Give wet waste to the Kachragadi vehicle for collection."
        ],
        "Hazardous Waste": [
            "Take to a designated hazardous waste collection center.",
            "Never mix hazardous waste with regular garbage.",
            "Seal and label hazardous waste before handing it over."
        ],
        "Electronic Waste": [
            "Return old devices to an authorized e-waste recycling center.",
            "Do not throw electronic items in normal bins.",
            "Check if manufacturers offer e-waste take-back programs."
        ],
        "Construction Waste": [
            "Hire a licensed waste removal service for disposal.",
            "Do not dump debris in open areas.",
            "Separate reusable construction material for recycling."
        ],
        "Biomedical Waste": [
            "Dispose of through authorized biomedical waste service providers.",
            "Never discard syringes or medicines in household bins.",
            "Use yellow-marked containers for biomedical waste."
        ]
    },
    "mr": {
        "Dry Waste": [
            "कागद, प्लास्टिक आणि धातू पुनर्चक्रण केंद्रात द्या.",
            "जैविक कोरडा कचरा असल्यास घरी कंपोस्ट करा.",
            "न पुनर्चक्रणयोग्य कोरडा कचरा कोरड्या डब्यात टाका."
        ],
        "Wet Waste": [
            "घरी किंवा नगरपालिका कम्पोस्टिंग सुविधेमध्ये कम्पोस्ट करा.",
            "अन्नाचे उरलेले तुकडे आणि सालांसाठी स्वयंपाकघर कंपोस्ट डब्बा वापरा.",
            "कचरा गाडीला ओला कचरा द्या."
        ],
        "Hazardous Waste": [
            "धोकादायक कचरा संकलन केंद्रात नेऊन द्या.",
            "धोकादायक कचरा सामान्य कचऱ्यात मिसळू नका.",
            "कचरा सुपूर्द करण्यापूर्वी तो सीलबंद आणि लेबल करा."
        ],
        "Electronic Waste": [
            "जुनी उपकरणे अधिकृत ई-कचरा पुनर्चक्रण केंद्रात द्या.",
            "इलेक्ट्रॉनिक वस्तू सामान्य कचरापेटीत टाकू नका.",
            "निर्माते ई-कचरा परत घेण्याची सेवा देतात का ते तपासा."
        ],
        "Construction Waste": [
            "परवानाधारक कचरा काढणी सेवा वापरा.",
            "कचरा किंवा मलबा उघड्या जागेत टाकू नका.",
            "पुनर्वापरयोग्य बांधकाम साहित्य वेगळे ठेवा."
        ],
        "Biomedical Waste": [
            "अधिकृत बायोमेडिकल कचरा सेवेद्वारेच निपटारा करा.",
            "सुई किंवा औषधे घरगुती कचऱ्यात टाकू नका.",
            "बायोमेडिकल कचऱ्यासाठी पिवळ्या चिन्हांकित डबे वापरा."
        ]
    }
}

# ========================
# 🔹 PROMPT TEMPLATE
# ========================
# DETECTION_PROMPT = """
# You are a professional AI waste detection expert.

# Analyze the image carefully and identify all visible waste items.

# If multiple objects of the same type (e.g. several banana peels or plastic bottles) are visible and close together,
# group them into a single detection entry.

# Each entry must include:
# {
#   "object": "descriptive item name (e.g. 'banana peel', 'set of banana peels', 'set of plastic bottles')",
#   "category": "one of ['Dry Waste', 'Wet Waste', 'Hazardous Waste', 'Electronic Waste', 'Construction Waste', 'Biomedical Waste']",
#   "bbox": [x1, y1, x2, y2]
# }

# ⚙️ STRICT RULES:
# 1. Group identical items (e.g. 5 banana peels → "set of banana peels").
# 2. Use a single bounding box that covers all grouped items.
# 3. If only one item exists, use its singular name (e.g. "banana peel").
# 4. Always include a valid bounding box in pixel coordinates [x1, y1, x2, y2].
# 5. Use only integer coordinates (no decimals).
# 6. Return VALID JSON ONLY — no markdown, no explanations, no extra text.

# Example output:
# [
#   {
#     "object": "set of banana peels",
#     "category": "Wet Waste",
#     "bbox": [120, 80, 400, 250]
#   },
#   {
#     "object": "plastic bottle",
#     "category": "Dry Waste",
#     "bbox": [420, 100, 520, 250]
#   }
# ]
# """

DETECTION_PROMPT = """
You are a precise AI waste-detection system.

Identify every visible waste item in the image and group identical nearby ones.

Return JSON only:
[
  {"object": "descriptive name", "category": "Dry Waste | Wet Waste | Hazardous Waste | Electronic Waste | Construction Waste | Biomedical Waste", "bbox": [x1, y1, x2, y2]}
]

Rules:
- Group similar items (e.g. many banana peels → "set of banana peels").
- Singular name if only one.
- Bounding box integers only.
- Detect small or partially visible waste.
- Classify correctly by material:
  * food, vegetables → Wet Waste
  * paper, plastic, metal → Dry Waste
  * electronics → Electronic Waste
  * cement, bricks → Construction Waste
  * masks, gloves, syringes → Biomedical Waste
  * chemicals, glass, paint → Hazardous Waste
Output valid JSON only.


 Example output:
[
  {
    "object": "set of banana peels",
    "category": "Wet Waste",
    "bbox": [120, 80, 400, 250]
  },
  {
    "object": "plastic bottle",
    "category": "Dry Waste",
    "bbox": [420, 100, 520, 250]
  }
]
"""


# ========================
# 🔹 HELPER FUNCTIONS
# ========================


def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]  # width, height


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
    volume = area_cm2 * depth_cm
    density = densities.get(category, 0.001)
    weight = density * volume
    max_weight = {
        "Dry Waste": 1.0, "Wet Waste": 0.8, "Hazardous Waste": 2.0,
        "Electronic Waste": 1.5, "Construction Waste": 5.0, "Biomedical Waste": 0.5
    }.get(category, 1.0)
    return min(weight, max_weight)


def aggregate_objects(results, lang="en"):
    grouped = defaultdict(list)
    for obj in results:
        grouped[(obj["object"], obj["category"])].append(obj)

    aggregated = []
    for (name, category), items in grouped.items():
        total_area = sum(o["area_cm2"] for o in items)
        total_weight = sum(o["tentative_weight_kg"] for o in items)
        disposal = DISPOSAL_GUIDE[lang].get(category, [])
        aggregated.append({
            "object": name,
            "category": category,
            "area_cm2": round(total_area, 1),
            "tentative_weight_kg": round(total_weight, 2),
            "disposal": disposal,
            "count": len(items),
            "color": "#{:02X}{:02X}{:02X}".format(*COLOR_MAP[category])
        })
    return aggregated


def classify_objects(image_path, lang="en"):
    try:
        w, h = get_image_dimensions(image_path)
        if not w:
            return []

        img = Image.open(image_path)
        model = genai.GenerativeModel(MODEL_ID)

        response = model.generate_content([DETECTION_PROMPT, img])
        text = response.text.strip()

        # clean JSON
        text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
        try:
            detections = json.loads(text)
        except Exception:
            print("⚠️ JSON parse error. Raw output:\n", text)
            return []

        validated = []
        for obj in detections:
            category = obj.get("category", "")
            if category not in WASTE_CATEGORIES:
                continue
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            pixel_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            area_cm2 = pixel_area * (0.0264583 ** 2)
            weight = estimate_weight(category, area_cm2)

            validated.append({
                "object": obj.get("object", "Unknown"),
                "category": category,
                "bbox": [x1, y1, x2, y2],
                "area_cm2": area_cm2,
                "tentative_weight_kg": weight
            })

        return aggregate_objects(validated, lang=lang)
    except Exception as e:
        print("Error:", e)
        return []


def draw_annotations(image_path, results, output_path="annotated_result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = COLOR_MAP[r["category"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = r["category"]
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(output_path, img)
    return output_path

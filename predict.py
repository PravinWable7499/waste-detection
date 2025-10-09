import os
import cv2
from PIL import Image
import google.generativeai as genai
import json
import re
from collections import defaultdict

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

DISPOSAL_GUIDE_EN = {
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
        "Never mix hazardous waste with regular household garbage.",
        "Seal and label hazardous waste before handing it over."
    ],
    "Electronic Waste": [
        "Return old devices to an authorized e-waste recycling center.",
        "Do not throw electronic items in normal bins.",
        "Check if manufacturers offer e-waste take-back programs."
    ],
    "Construction Waste": [
        "Hire a licensed waste removal service for disposal.",
        "Do not dump debris or rubble in open areas.",
        "Separate reusable construction material for recycling."
    ],
    "Biomedical Waste": [
        "Dispose of through authorized biomedical waste service providers.",
        "Never discard syringes or medicines in household bins.",
        "Use yellow-marked containers for biomedical waste segregation."
    ]
}

DISPOSAL_GUIDE_MR = {
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

DETECTION_PROMPT = """
You are a professional AI-powered waste classification expert with 10+ years of experience in environmental science.

Your task: Analyze the provided image and identify EVERY visible waste item with high precision.

✅ STRICT INSTRUCTIONS:
1. Detect ALL waste items — even small, partial, or overlapping ones.
2. For EACH detected object, provide:
   - "object": A clear, descriptive name (e.g., "banana peel", "plastic bottle cap", "crumpled paper")
   - "category": One of these EXACT categories:
        "Dry Waste", "Wet Waste", "Hazardous Waste", "Electronic Waste", "Construction Waste", "Biomedical Waste"
   - "bbox": [x1, y1, x2, y2] — pixel coordinates (top-left to bottom-right)
3. NEVER omit any visible waste item.
4. If uncertain about category, choose the MOST LIKELY one based on visual cues.
5. Return ONLY valid JSON — no explanations, no extra text.

⚠️ CRITICAL RULES:
- Do NOT return empty arrays or null values.
- Do NOT combine multiple objects into one.
- Do NOT skip partially visible items.
- Use integer pixel coordinates only (no decimals).
Example:
[
  {
    "object": "banana peel",
    "category": "Wet Waste",
    "bbox": [100, 50, 300, 200]
  }
]
"""

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h

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
        "Dry Waste": 1.0,
        "Wet Waste": 0.8,
        "Hazardous Waste": 2.0,
        "Electronic Waste": 1.5,
        "Construction Waste": 5.0,
        "Biomedical Waste": 0.5
    }.get(category, 1.0)
    return min(weight, max_weight)

def aggregate_objects(results, lang='en'):
    from collections import defaultdict
    grouped = defaultdict(list)
    for obj in results:
        key = (obj["object"], obj["category"])
        grouped[key].append(obj)
    
    aggregated = []
    for (name, category), items in grouped.items():
        total_area = sum(obj.get("area_cm2", 0) for obj in items)
        total_weight = sum(obj.get("tentative_weight_kg", 0) for obj in items)
        count = len(items)
        first = items[0]

        disposal = DISPOSAL_GUIDE_MR.get(category, []) if lang == 'mr' else DISPOSAL_GUIDE_EN.get(category, [])

        aggregated.append({
            "object": name,
            "category": category,
            "bbox": first["bbox"],
            "disposal": disposal,
            "area_cm2": round(total_area, 1),
            "tentative_weight_kg": round(total_weight, 2),
            "count": count
        })
    return aggregated

def classify_objects(image_path, lang='en'):
    try:
        img_width, img_height = get_image_dimensions(image_path)
        print("Image dimensions:", img_width, img_height)

        if img_width is None:
            return []

        img_pil = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        prompt = f"{DETECTION_PROMPT}\n\nImage dimensions: {img_width} × {img_height} pixels"

        response = model.generate_content([prompt, img_pil])
        print("response ",response)
        text = response.text.strip()
        print("text ",text)

        cleaned = re.sub(r'^```(?:json)?\s*\n', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n\s*```\s*$', '', cleaned).strip()
        parsed=[]
        try:
            parsed = json.loads(cleaned)
            print("parsed ",parsed)
        except json.JSONDecodeError:
            print("parsed ")

            return []
        



        validated = []
        print("parsed = ",parsed)

        for obj in parsed:
            category = obj.get("category", "").strip()
            bbox = obj.get("bbox", [])
            obj_name = obj.get("object", "Unknown")

            if category not in WASTE_CATEGORIES or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            pixel_area = (x2 - x1) * (y2 - y1)
            area_cm2 = pixel_area * (0.0264583 ** 2)
            weight_kg = estimate_weight(category, area_cm2)

            validated.append({
                "object": obj_name,
                "category": category,
                "bbox": [x1, y1, x2, y2],
                "area_cm2": area_cm2,
                "tentative_weight_kg": weight_kg
            })
        dataAggregat=aggregate_objects(validated, lang=lang)
        print("dataAggregat ",dataAggregat)

        return dataAggregat

    except Exception as e:
        print(f"Error in classify_objects: {e}")
        return []

def draw_annotations(image_path, results, output_path="annotated_result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None

    for obj in results:
        x1, y1, x2, y2 = obj["bbox"]
        color = COLOR_MAP[obj["category"]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 6, color, -1)
        label = f"{obj['category']}"
        cv2.putText(img, label, (cx, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    return output_path


def translate_to_marathi(text: str) -> str:
    """Translate English text to Marathi using Gemini"""
    try:
        # Use the same model as detection
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Translate to Marathi: '{text}'. Return ONLY the translation."
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 20}
        )
        result = response.text.strip().strip('".\'')
        return result if result else text
    except Exception as e:
        print(f"Translation error for '{text}': {str(e)}")
        return text  # Keep original if fails
# prediction.py
import os
import cv2
from PIL import Image
import google.generativeai as genai
import json
import re

# --- CONFIGURE API KEY ---
GOOGLE_API_KEY = 'AIzaSyDTjKDxjZK02raiHrzYbAlGu1n1lM-ptag'  # Replace with your key
genai.configure(api_key=GOOGLE_API_KEY) 


# --- WASTE CATEGORIES & COLORS ---
WASTE_CATEGORIES = [
    "Dry Waste",
    "Wet Waste",
    "Hazardous Waste",
    "Electronic Waste",
    "Construction Waste",
    "Biomedical Waste"
]

COLOR_MAP = {
    "Dry Waste": (0, 255, 0),      # Green
    "Wet Waste": (0, 255, 255),    # Yellow
    "Hazardous Waste": (0, 0, 255), # Red
    "Electronic Waste": (255, 0, 255), # Magenta
    "Construction Waste": (255, 255, 0), # Cyan
    "Biomedical Waste": (128, 0, 128)   # Purple
}

# --- DISPOSAL GUIDELINES ---
DISPOSAL_GUIDE = {
    "Dry Waste": "📦 Recycle or compost if biodegradable. Place in dry waste bin.",
    "Wet Waste": "🍃 Compost at home or in municipal composting facility.",
    "Hazardous Waste": "⚠️ Take to designated hazardous waste collection center. Do not throw in regular trash.",
    "Electronic Waste": "🔋 Return to e-waste recycling center. Never dispose in landfill.",
    "Construction Waste": "🏗️ Hire a licensed waste removal service. Do not dump illegally.",
    "Biomedical Waste": "🩺 Dispose only through medical waste services. Never discard in household bins."
}

# --- IMPROVED PROMPT WITH DISPOSAL SUGGESTIONS ---
DETECTION_PROMPT = """
You are a professional waste classification expert specializing in environmental sustainability.

 IMPORTANT INSTRUCTIONS:
- Analyze the image carefully.
- Identify ALL visible waste items with PRECISE bounding boxes.
- Each object must be classified accurately based on its physical properties.
- Every object must have its own tight bounding box around only that object.

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
    """Get image width and height"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height

def classify_objects(image_path):
    """Detect and classify waste objects in image"""
    try:
        img_width, img_height = get_image_dimensions(image_path)
        if img_width is None:
            return []

        img_pil = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        dimension_prompt = f"{DETECTION_PROMPT}\n\nImage dimensions: {img_width} × {img_height} pixels"

        response = model.generate_content([dimension_prompt, img_pil])
        text = response.text.strip()

        # Clean JSON output
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

            # Clamp to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))

            # Default disposal if missing
            if not disposal:
                disposal = DISPOSAL_GUIDE.get(category, "Dispose responsibly in appropriate bin.")

            validated_results.append({
                "object": obj_name,
                "category": category,
                "bbox": [x1, y1, x2, y2],
                "disposal": disposal
            })

        return validated_results

    except Exception as e:
        print(f"Error during classification: {e}")
        return []

def estimate_weight(category, area_cm2):
    """Estimate tentative weight (kg) based on category and area"""
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
    """Draw bounding boxes and labels on image, save to output_path"""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return None

    height, width = img_cv.shape[:2]

    for obj in results:
        category = obj["category"]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = bbox

        color = COLOR_MAP[category]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

        # Label: Only show waste type (e.g., "Dry Waste")
        label = f"{category}"
        font_scale = 0.7
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above box
        label_y = y1 - 10 if y1 > text_h + 10 else y2 + 20
        label_x = x1

        if label_x + text_w > width:
            label_x = width - text_w - 5

        # Draw BLACK semi-transparent background behind text
        overlay = img_cv.copy()
        cv2.rectangle(overlay, 
                     (label_x - 5, label_y - text_h - 5), 
                     (label_x + text_w + 5, label_y + 5), 
                     (0, 0, 0), -1)  # Black background
        cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)

        # Draw WHITE text on top
        cv2.putText(img_cv, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness)

        # Center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img_cv, (cx, cy), 4, color, -1)

    cv2.imwrite(output_path, img_cv)
    return output_path

# test_prediction.py
from predict import classify_objects, estimate_weight

def format_detection_results(results, image_path):
    """Format detection results for terminal output"""
    
    # Get image dimensions for area calculation
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(" Could not load image.")
        return
    
    height, width = img.shape[:2]
    px_to_cm = 0.1  # 100 pixels = 10 cm

    print("\n" + "="*80)
    print("🗑️  WASTE DETECTION RESULTS")
    print("="*80 + "\n")

    for i, obj in enumerate(results, 1):
        # Calculate area
        bbox = obj['bbox']
        area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_cm2 = area_px * (px_to_cm ** 2)
        
        # Format category color (for terminal)
        category = obj['category']
        color_code = {
            "Dry Waste": "\033[32m",      # Green
            "Wet Waste": "\033[33m",      # Yellow
            "Hazardous Waste": "\033[31m", # Red
            "Electronic Waste": "\033[35m", # Magenta
            "Construction Waste": "\033[36m", # Cyan
            "Biomedical Waste": "\033[35m"  # Purple
        }.get(category, "\033[0m")
        
        # Build output string
        output = f"{i}. {obj['object']}\n"
        output += f"🗃️  Category: {color_code}{category}\033[0m\n"
        output += f"📐  Area: {area_cm2:.1f} cm²"
        
        # Add weight only for non-Hazardous/Construction waste
        if category not in ["Hazardous Waste", "Construction Waste"]:
            weight_kg = estimate_weight(category, area_cm2)
            output += f" ⚖️  Tentative Weight: {weight_kg:.2f} kg"
        
        output += f"\n🧹  How to Dispose: {obj['disposal']}\n"
        
        print(output)
        if i < len(results):
            print("-" * 50)

def main():
    #  CONFIGURE INPUT/OUTPUT PATHS
    input_image = "Frame 10 (1).png"  # Replace with your image path
    output_image = "annotated_result1.jpg"

    print("🔍 Analyzing image...")
    
    # Detect objects
    results = classify_objects(input_image)
    
    if not results:
        print(" No waste objects detected.")
        return
    
    # Draw annotations
    from predict import draw_annotations
    output_path = draw_annotations(input_image, results, output_image)
    
    if output_path:
        print(f" Annotated image saved to: {output_path}")
    else:
        print(" Failed to save annotated image.")
    
    # Format and display results
    format_detection_results(results, input_image)

if __name__ == "__main__":
    main()
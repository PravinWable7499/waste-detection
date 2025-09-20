# Waste Detection System

This is an Automatic Waste Detection and Classification System built using Python, FastAPI, and HTML/CSS/JS.

## 🚀 Features
- Upload image and get waste detection results
- Annotated image download option
- Multi-language support (English + Marathi)

## 📂 Project Structure
- **main.py** → Entry point of the application
- **predict.py** → Waste detection logic
- **templates/** → HTML pages (`index.html`, `result.html`)
- **static/** → CSS, JS, and logo files
- **uploads/** → Stores uploaded images

## 🛠 Setup
```bash
git clone https://github.com/your-username/waste-detection.git
cd waste-detection
python -m venv venv
venv/Scripts/activate  # (Windows)
pip install -r requirements.txt
uvicorn main:app --reload

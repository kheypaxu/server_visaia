from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import io
import requests
import base64
import os

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model URLs on HuggingFace
# -----------------------------
CACHE_DIR = "/tmp/models"
os.makedirs(CACHE_DIR, exist_ok=True)

MOBILENET_URL = "https://huggingface.co/xelpaxu05/faw_models/resolve/main/mobilenetv3_faw_vs_notfaw.pth"
YOLO_URL = "https://huggingface.co/xelpaxu05/faw_models/resolve/main/best.pt"

def download_model(url, filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} from HuggingFace...")
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path

# -----------------------------
# Load MobileNetV3 (FAW vs NotFAW)
# -----------------------------
mobilenet_path = download_model(MOBILENET_URL, "mobilenetv3_faw_vs_notfaw.pth")
mobilenet_model = models.mobilenet_v3_small(pretrained=False)
mobilenet_model.classifier[3] = torch.nn.Linear(
    mobilenet_model.classifier[3].in_features, 2
)
mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location=DEVICE))
mobilenet_model.to(DEVICE)
mobilenet_model.eval()

CLASS_NAMES = ["FAW", "NotFAW"]
mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Load YOLO FAW life stage model
# -----------------------------
yolo_path = download_model(YOLO_URL, "best.pt")
yolo_model = YOLO(yolo_path)
FAW_CLASSES = ["egg", "larva", "pupa", "moth"]

# -----------------------------
# FAW Life Stage Risk Rules
# -----------------------------
LIFE_STAGE_RISK = {
    "egg": "Low",
    "larva": "High",
    "pupa": "Low",
    "moth": "High"
}

# -----------------------------
# Helper: GPT-OSS API via OpenRouter (NotFAW)
# -----------------------------
def ask_llm(image_bytes):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: API Key missing."

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    models_to_try = [
        "google/gemma-3-27b-it:free",
        "qwen/qwen-2-vl-7b-instruct:free",
        "nvidia/llama-3.2-nv-vision-70b:free"
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "FAW Detection App",
    }

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify this pest: common and scientific name."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 429:
                continue
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except:
            continue

    return "AI analysis unavailable; all models are busy."

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # --- Step 1: FAW vs NotFAW ---
        input_tensor = mobilenet_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = mobilenet_model(input_tensor)
            _, pred = torch.max(outputs, 1)
            class_prediction = CLASS_NAMES[pred.item()]

        if class_prediction == "FAW":
            # --- Step 2: YOLO Life Stage Detection ---
            results = yolo_model(image)
            detected_classes = []
            boxes_data = []
            width, height = image.size

            for result in results:
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    stage_name = FAW_CLASSES[cls_idx]
                    detected_classes.append(stage_name)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes_data.append({
                        "class": stage_name,
                        "x": x1 / width,
                        "y": y1 / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height
                    })

            stage_text = ", ".join(detected_classes) if detected_classes else "no life stage detected"
            pest_name = "Fall Army Worm"
            full_prediction = f"{pest_name} - {stage_text} Stage"

            # --- Step 3: Compute Risk Level ---
            stage_risks = [LIFE_STAGE_RISK.get(stage, "Unknown") for stage in detected_classes]
            overall_risk = "Low"
            if "High" in stage_risks:
                overall_risk = "High"
            elif "Unknown" in stage_risks:
                overall_risk = "Unknown"

            # --- Step 4: Rule-Based Explanation ---
            explanation = []
            for stage, risk in zip(detected_classes, stage_risks):
                explanation.append(f"{stage.capitalize()} stage → {risk} risk of spread")
            explanation_text = "; ".join(explanation) if explanation else "No life stages detected, risk cannot be assessed."

            return jsonify({
                "pest": pest_name,
                "stage": stage_text,
                "full_prediction": full_prediction,
                "boxes": boxes_data,
                "risk_level": overall_risk,
                "explanation": explanation_text
            })

        else:
            # --- Not FAW: Use AI analysis ---
            ai_analysis = ask_llm(img_bytes)
            return jsonify({
                "pest": "Unknown/Other Pest",
                "stage": "",
                "full_prediction": ai_analysis,
                "boxes": [],
                "risk_level": "",
                "explanation": ai_analysis
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()

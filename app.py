import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

# =====================
# BASIC CONFIG
# =====================
MODEL_PATH = "model/ct_effnet_best.pth"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔌 Device:", DEVICE)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================
# LOAD DISEASE INFO JSON (flat structure expected)
# =====================
with open("diseases.json", "r", encoding="utf-8") as f:
    raw_disease_info = json.load(f)

# Build normalized lookup: lower_key -> (orig_key, info_dict)
DISEASE_INFO = {}
for orig_key, info in raw_disease_info.items():
    norm = orig_key.strip().lower()
    DISEASE_INFO[norm] = {"orig_key": orig_key, "info": info}

# =====================
# OPTIONAL LABEL_MAP (if your model sometimes predicts categories like "Brain")
# Make sure keys and disease names are normalized for lookup
# =====================
LABEL_MAP = {
    "pulmonary arteries": {
        "category": "Pulmonary Conditions",
        "diseases": [
            "subsegmental pulmonary embolism",
            "acute pulmonary embolism",
            "central pulmonary embolism",
            "chronic pulmonary embolism",
            "chronic thromboembolic pulmonary hypertension",
            "in situ pulmonary artery thrombosis",
            "normal (pulmonary arteries – pulmonary embolism)",
            "pulmonary infarction",
            "recurrent pulmonary embolism",
            "saddle pulmonary embolism",
            "segmental pulmonary embolism"
        ]
    },
    "lung nodules": {
        "category": "Lung Nodule Conditions",
        "diseases": [
            "adenocarcinoma in situ (ais)",
            "benign granuloma (healed tb, fungal)",
            "ground-glass nodule (ggn)",
            "minimally invasive adenocarcinoma (mia)",
            "normal (lung nodules – small lesions)",
            "part-solid lung nodule",
            "pulmonary metastatic nodules",
            "pulmonary tuberculoma",
            "solitary pulmonary nodule (spn)",
            "subsolid lung nodule"
        ]
    },
    "intracranial hemorrhage": {
        "category": "Brain Hemorrhagic Conditions",
        "diseases": [
            "cerebral contusion with hemorrhage",
            "chronic subdural hematoma",
            "coagulopathy-related intracranial hemorrhage",
            "diffuse axonal injury with hemorrhagic foci",
            "early intracerebral hemorrhage (small ich)",
            "hemorrhagic transformation of ischemic stroke",
            "normal (intracranial hemorrhage)",
            "subacute subdural hematoma",
            "traumatic intracerebral hemorrhage (tich)"
        ]
    },
    "brain": {
        "category": "Other Brain Conditions",
        "diseases": [
            "cerebral amyloid angiopathy-related intracerebral hemorrhage",
            "hypoxic-ischemic brain injury",
            "lacunar infarct",
            "metabolic encephalopathy",
            "neurocysticercosis",
            "normal (brain parenchyma)",
            "small metastatic brain lesions",
            "small primary brain tumor",
            "traumatic intraparenchymal hemorrhage",
            "brain edema",
            "cerebral contusion",
            "cerebritis",
            "diffuse axonal injury (dai)",
            "early ischemic stroke",
            "encephalitis"
        ]
    }
}
# normalize LABEL_MAP keys and disease strings
LABEL_MAP = {k.lower(): {"category": v["category"], "diseases": [d.lower() for d in v["diseases"]]} for k, v in LABEL_MAP.items()}


# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# classes in checkpoint: ensure we preserve raw strings but also prepare normalized list
class_names_raw = checkpoint.get("classes", [])
class_names = [c.strip() for c in class_names_raw]
class_names_norm = [c.strip().lower() for c in class_names_raw]

num_classes = len(class_names)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded with classes:", class_names)

# =====================
# IMAGE TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================
# HELPERS
# =====================
def field_to_list(v):
    """Convert a JSON field to a list for display (Symptoms/Diagnosis/Medicines/Prevention/Care).
       Accepts list or comma/semicolon separated string."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        # split on common delimiters
        parts = []
        for seg in v.split("\n"):
            parts.extend(seg.split(";"))
        final = []
        for seg in ",".join(parts).split(","):
            item = seg.strip()
            if item:
                final.append(item)
        return final
    # fallback
    return [str(v)]

def find_info_for_label(raw_label):
    """
    Attempt to find disease info for predicted raw_label using several strategies:
      1) exact match (normalized) against DISEASE_INFO keys
      2) if raw_label is a known category in LABEL_MAP -> pick first disease listed that exists in DISEASE_INFO
      3) strip parentheses (e.g. "Brain (Parenchyma)") and retry
      4) substring match against DISEASE_INFO keys
    Returns (display_name, info_dict) or (None, None)
    """
    if not raw_label:
        return None, None

    label_norm = raw_label.strip().lower()

    # 1) exact disease key
    if label_norm in DISEASE_INFO:
        rec = DISEASE_INFO[label_norm]
        return rec["orig_key"], rec["info"]

    # 2) label is category from LABEL_MAP
    if label_norm in LABEL_MAP:
        for candidate in LABEL_MAP[label_norm]["diseases"]:
            if candidate in DISEASE_INFO:
                rec = DISEASE_INFO[candidate]
                return rec["orig_key"], rec["info"]

    # 3) strip parentheses and retry
    if "(" in label_norm:
        parent = label_norm.split("(", 1)[0].strip()
        if parent in DISEASE_INFO:
            rec = DISEASE_INFO[parent]
            return rec["orig_key"], rec["info"]
        if parent in LABEL_MAP:
            for candidate in LABEL_MAP[parent]["diseases"]:
                if candidate in DISEASE_INFO:
                    rec = DISEASE_INFO[candidate]
                    return rec["orig_key"], rec["info"]

    # 4) substring matching heuristics
    for k, rec in DISEASE_INFO.items():
        if label_norm in k or k in label_norm:
            return rec["orig_key"], rec["info"]

    # not found
    return None, None

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"})

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        # Preprocess & predict
        img = Image.open(save_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        raw_label = class_names[pred_idx.item()] if pred_idx.item() < len(class_names) else str(pred_idx.item())
        confidence = float(conf.item()) * 100.0

        display_name, info = find_info_for_label(raw_label)

        # If not found, return a helpful default
        if info is None:
            info = {
                "Description": f"Information for '{raw_label}' is being updated.",
                "Symptoms": "",
                "Diagnosis": "",
                "Medicines": "",
                "Prevention": "",
                "Care": ""
            }
            display_name = raw_label

        # Normalize fields for frontend
        response_info = {
            "Description": info.get("Description", ""),
            "Symptoms": field_to_list(info.get("Symptoms", "")),
            "Diagnosis": field_to_list(info.get("Diagnosis", "")),
            "Medicines": field_to_list(info.get("Medicines", "")),
            "Prevention": field_to_list(info.get("Prevention", "")),
            "Care": field_to_list(info.get("Care", ""))
        }

        return jsonify({
            "disease": display_name,
            "model_label": raw_label,
            "confidence": round(confidence, 2),
            "image": file.filename,
            "info": response_info
        })

    except Exception as e:
        # return the error so you can debug quickly
        return jsonify({"error": str(e)}), 500

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)

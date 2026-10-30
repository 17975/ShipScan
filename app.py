"""
app.py

Flask web application for ShipScan:
 - Serves an upload page (templates/index.html)
 - Accepts an uploaded image, runs it through a model, and returns a prediction
 - Keeps a simple in-memory "pantry log" of recognized items

MODEL LOADING BEHAVIOUR (automatic):
 - If model/food_classifier.h5 exists (i.e. you have run train_model.py on your
   own collected dataset), the app uses YOUR custom-trained model.
 - A pretrained image/text model recognizes common foods and household items,
   and rejects images that look unrelated.

IMPORTANT (read this before submitting your assignment):
   The custom model is trained on your own collected dataset. For best accuracy
   on any food, add varied photos of it to dataset/<food>/ and retrain.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import hashlib
import json
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import open_clip
import torch

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get(
        "CORS_ALLOWED_ORIGIN", "*"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


MODEL_PATH = os.path.join("model", "food_classifier.h5")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")
IMG_SIZE = (224, 224)

FOOD_LABELS = (
    "apple", "banana", "orange", "lemon", "lime", "grapefruit", "strawberry",
    "blueberry", "raspberry", "blackberry", "grapes", "mango", "pineapple",
    "watermelon", "cantaloupe", "pear", "peach", "plum", "kiwi", "pomegranate",
    "avocado", "coconut", "cherry", "fig", "papaya", "tomato", "carrot",
    "onion", "red onion", "yellow onion", "white onion", "shallot", "green onion",
    "leek", "garlic", "potato",
    "sweet potato", "cucumber", "bell pepper", "chili pepper", "broccoli",
    "cauliflower", "cabbage", "lettuce", "spinach", "kale", "celery", "corn",
    "mushroom", "zucchini", "eggplant", "pumpkin", "butternut squash",
    "green beans", "peas", "beet", "radish", "ginger", "asparagus",
    "brussels sprouts", "bread", "bagel", "tortilla", "rice", "pasta", "noodles",
    "oatmeal", "cereal", "beans", "lentils", "chickpeas", "egg", "milk", "cheese",
    "yogurt", "butter", "chicken", "beef", "pork", "fish", "salmon", "tuna",
    "shrimp", "bacon", "sausage", "pizza", "sandwich", "hamburger", "hot dog",
    "soup", "salad", "french fries", "potato chips", "olive oil", "flour", "sugar",
    "salt", "honey", "tea", "coffee", "chocolate", "cake", "cookie", "donut",
    "ice cream", "canned food", "food package",
)
HOUSEHOLD_LABELS = (
    "bottle", "cup", "mug", "glass", "plate", "bowl", "fork", "spoon",
    "kitchen knife", "cooking pot", "frying pan", "kettle", "toaster",
    "microwave oven", "refrigerator", "blender", "coffee maker", "dish soap",
    "cleaning spray", "laundry detergent", "sponge", "broom", "mop",
    "vacuum cleaner", "trash can", "chair", "table", "sofa", "bed", "lamp",
    "book", "phone", "laptop", "television", "remote control", "clothing",
    "shoe", "towel", "toothbrush", "shampoo bottle", "medicine bottle",
    "cardboard box", "basket", "keys", "scissors", "hand tool", "flower pot",
    "house plant",
)
NON_HOUSEHOLD_LABELS = (
    "dog", "cat", "horse", "bird", "cow", "sheep", "rabbit", "squirrel",
    "wild animal", "person", "car", "truck", "bicycle", "motorcycle", "airplane",
    "boat", "street scene", "mountain landscape", "ocean scene", "sports field",
    "toy", "video game", "computer screenshot", "document", "painting",
)
MIN_GENERAL_TARGET_SCORE = 0.20
MIN_GENERAL_FOOD_MARGIN = 0.015
MIN_GENERAL_HOUSEHOLD_MARGIN = 0.005
MIN_GENERAL_NON_TARGET_MARGIN = 0.005

USING_CUSTOM_MODEL = os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)

if USING_CUSTOM_MODEL:
    print("[ShipScan] Custom trained model found - using food_classifier.h5")
    custom_model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
else:
    print("[ShipScan] No custom model found yet - using general food and household recognition "
          "only. Train your own model with "
          "train_model.py to add your collected classes.")
    custom_model = None
    class_names = []

food_labels = tuple(dict.fromkeys(
    (*FOOD_LABELS, *(name.replace("_", " ").lower() for name in class_names))
))
food_prompts = tuple(f"a photo of {label}" for label in food_labels)

general_model, _, general_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained=None, device="cpu"
)
general_checkpoint_url = open_clip.get_pretrained_cfg("ViT-B-32", "openai")["url"]
general_checkpoint_path = os.path.join(
    torch.hub.get_dir(), "checkpoints", "clip_vit_b_32_openai.pt"
)
general_checkpoint_hash = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
if not os.path.exists(general_checkpoint_path):
    os.makedirs(os.path.dirname(general_checkpoint_path), exist_ok=True)
    torch.hub.download_url_to_file(
        general_checkpoint_url,
        general_checkpoint_path,
        hash_prefix=general_checkpoint_hash[:8],
    )
checkpoint_digest = hashlib.sha256()
with open(general_checkpoint_path, "rb") as checkpoint_file:
    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
        checkpoint_digest.update(chunk)
if checkpoint_digest.hexdigest() != general_checkpoint_hash:
    raise RuntimeError(
        f"Invalid OpenCLIP checkpoint at {general_checkpoint_path}; "
        "remove the incomplete file and restart to download it again."
    )
# The hash-verified official OpenAI checkpoint is a TorchScript archive.
open_clip.load_checkpoint(
    general_model, general_checkpoint_path, weights_only=False
)
general_model.eval()
general_tokenizer = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    food_text = general_tokenizer(food_prompts)
    food_text_features = general_model.encode_text(food_text)
    food_text_features /= food_text_features.norm(dim=-1, keepdim=True)
    household_text = general_tokenizer(
        tuple(f"a photo of {label}" for label in HOUSEHOLD_LABELS)
    )
    household_text_features = general_model.encode_text(household_text)
    household_text_features /= household_text_features.norm(dim=-1, keepdim=True)
    non_household_text = general_tokenizer(
        tuple(f"a photo of {label}" for label in NON_HOUSEHOLD_LABELS)
    )
    non_household_text_features = general_model.encode_text(non_household_text)
    non_household_text_features /= non_household_text_features.norm(dim=-1, keepdim=True)

# Simple in-memory pantry log: [{"item": str, "confidence": float, "time": str}]
pantry_log = []


def prepare_image(file_storage):
    """Load an uploaded image file into a model-ready array."""
    img = Image.open(io.BytesIO(file_storage.read()))
    img = ImageOps.exif_transpose(img)  # fix sideways phone photos
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_with_custom_model(img_array):
    predictions = custom_model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])
    label = class_names[top_idx]
    all_confidences = {
        class_names[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(class_names))
    }
    return label, confidence, all_confidences


def predict_with_general_item_model(file_storage):
    image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(file_storage))
    ).convert("RGB")
    image_tensor = general_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = general_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        def rank_candidates(labels, text_features):
            scores = (100.0 * image_features @ text_features.T)[0]
            probabilities = scores.softmax(dim=-1)
            top_indices = torch.topk(scores, k=min(5, len(labels))).indices.tolist()
            return [
                {
                    "label": labels[index],
                    "confidence": round(float(probabilities[index]) * 100, 2),
                    "similarity": float(scores[index] / 100.0),
                }
                for index in top_indices
            ]

        return {
            "food": rank_candidates(food_labels, food_text_features),
            "household": rank_candidates(HOUSEHOLD_LABELS, household_text_features),
            "other": rank_candidates(NON_HOUSEHOLD_LABELS, non_household_text_features),
        }


@app.route("/")
def index():
    return render_template(
        "index.html",
        pantry_log=pantry_log,
        using_custom_model=USING_CUSTOM_MODEL,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        img_array = prepare_image(io.BytesIO(image_bytes))
        candidate_groups = predict_with_general_item_model(image_bytes)
        general_food_matches = candidate_groups["food"]
        household_matches = candidate_groups["household"]
        other_matches = candidate_groups["other"]
        food_candidate = general_food_matches[0]
        household_candidate = household_matches[0]
        other_candidate = other_matches[0]

        def is_confident(matches, margin):
            return (
                matches[0]["similarity"] >= MIN_GENERAL_TARGET_SCORE
                and matches[0]["similarity"] - matches[1]["similarity"]
                >= margin
            )

        food_is_confident = is_confident(
            general_food_matches, MIN_GENERAL_FOOD_MARGIN
        )
        household_is_confident = is_confident(
            household_matches, MIN_GENERAL_HOUSEHOLD_MARGIN
        )
        best_target_similarity = max(
            food_candidate["similarity"] if food_is_confident else 0,
            household_candidate["similarity"] if household_is_confident else 0,
        )
        if (
            other_candidate["similarity"] >= MIN_GENERAL_TARGET_SCORE
            and other_candidate["similarity"]
            >= best_target_similarity + MIN_GENERAL_NON_TARGET_MARGIN
        ) or best_target_similarity == 0:
            return jsonify({
                "label": "This is not a household item or food.",
                "item_type": "unsupported",
                "is_valid": False,
                "confidence": 0,
                "all_confidences": {},
                "additional_food_matches": [],
                "additional_household_matches": [],
                "using_custom_model": USING_CUSTOM_MODEL,
                "recognition_source": "general",
            })

        item_type = (
            "household"
            if household_is_confident
            and (
                not food_is_confident
                or household_candidate["similarity"] > food_candidate["similarity"]
            )
            else "food"
        )

        if item_type == "household":
            label = household_candidate["label"]
            confidence = household_candidate["confidence"] / 100
            recognition_source = "general"
            all_confidences = {
                match["label"]: match["confidence"]
                for match in household_matches
            }
            additional_food_matches = []
            additional_household_matches = household_matches[1:]
        elif USING_CUSTOM_MODEL:
            label, confidence, all_confidences = predict_with_custom_model(img_array)
            recognition_source = "custom"
            normalized_class_names = {
                name.replace("_", " ").lower() for name in class_names
            }
            custom_candidate_is_known = (
                food_candidate["label"] in normalized_class_names
            )
            general_candidate_is_confident = food_is_confident
            if not custom_candidate_is_known and general_candidate_is_confident:
                label = food_candidate["label"]
                confidence = food_candidate["confidence"] / 100
                recognition_source = "general"

            additional_food_matches = [
                {"label": match["label"], "confidence": match["confidence"]}
                for match in general_food_matches
                if match["label"] != label
            ]
            additional_household_matches = []
        else:
            label = food_candidate["label"]
            confidence = food_candidate["confidence"] / 100
            recognition_source = "general"
            all_confidences = {
                match["label"]: match["confidence"]
                for match in general_food_matches
            }
            additional_food_matches = general_food_matches[1:]
            additional_household_matches = []

        # Log the recognized item
        entry = {
            "item": label,
            "item_type": item_type,
            "confidence": round(confidence * 100, 2),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # The phone's auto-scan mode sends log=0 so it doesn't spam the pantry log
        if request.form.get("log", "1") == "1":
            pantry_log.append(entry)

        # Optional: simple "low stock" style logic placeholder
        # (count how many times this item has been logged recently)
        count_recent = sum(1 for e in pantry_log if e["item"] == label)

        return jsonify({
            "label": label,
            "item_type": item_type,
            "is_valid": True,
            "confidence": entry["confidence"],
            "logged_count": count_recent,
            "all_confidences": all_confidences,
            "additional_food_matches": additional_food_matches,
            "additional_household_matches": additional_household_matches,
            "using_custom_model": USING_CUSTOM_MODEL,
            "recognition_source": recognition_source,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pantry_log")
def get_pantry_log():
    return jsonify(pantry_log)


@app.route("/clear_log", methods=["POST"])
def clear_log():
    pantry_log.clear()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    """Report whether the prediction model is available."""
    return jsonify({
        "status": "ok",
        "using_custom_model": USING_CUSTOM_MODEL,
        "using_general_food_model": True,
    })


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Image upload is too large (maximum 12 MB)."}), 413


def get_lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ip = get_lan_ip()
    port = int(os.environ.get("PORT", "5000"))
    print("\n" + "=" * 60)
    print("  ShipScan server running")
    print(f"  Browser (this computer): http://127.0.0.1:{port}")
    print(f"  PHONE (Expo Go) server address:  http://{ip}:{port}")
    print("  Phone and computer must be on the same Wi-Fi.")
    print("=" * 60 + "\n")
    # host=0.0.0.0 lets your phone reach this server; debug off so the
    # model isn't loaded twice
    app.run(host="0.0.0.0", port=port, debug=False)

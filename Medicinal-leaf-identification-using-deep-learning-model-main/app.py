import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import logging, warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
from PIL import Image
import io, base64
import mysql.connector
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "change_this_for_production"

#  Path to your .keras model
MODEL_FILENAME = r"C:\Users\91843\Downloads\best_efficientnetb0_leaf.keras"

#  Medicinal leaf classes
CLASS_NAMES = [
    "Aloevera","Amla","Amruthaballi","Bamboo","Beans","Bringaraja","Castor","Chilly",
    "Coffee","Coriander","Curry","Doddapathre","Drumstick","Ginger","Guava","Hibiscus",
    "Honge","Jackfruit","Jasmine","Kamakasturi","Lemon","Lemongrass","Mango","Marigold",
    "Mint","Neem","Nerale","Onion","Palak(Spinach)","Papaya","Parijatha","Pepper",
    "Pomegranate","Pumpkin","Raddish","Rose","Sampige","Seethapala","Tamarind","Thumbe",
    "Tomato","Tulsi","Turmeric"
]

_model = None


# LOAD MODEL (Keras 3 / .keras support)

def load_model_once():
    global _model
    if _model is None:
        print(f">>> Loading model from: {MODEL_FILENAME}")
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"Model not found at: {MODEL_FILENAME}")

        _model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
        print(">>> Model loaded successfully\n")

    return _model



# IMAGE PREPROCESSING

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    # EfficientNet preprocessing
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return np.expand_dims(arr, axis=0)



# DATABASE FETCH FIXED

def get_db_info(class_name):
    """
    FIXED VERSION:
    - Uses dictionary=True
    - Consumes results properly
    - 'Unread result found' eliminated
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="medicinal_leaf_db"
        )
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT * FROM leaves WHERE class_name = %s", (class_name,))

        # fetchone MUST be called before closing cursor
        row = cur.fetchone()

        # clean close to avoid unread result error
        cur.fetchall()
        cur.close()
        conn.close()

        return row

    except mysql.connector.errors.InterfaceError as e:
        print("MySQL InterfaceError:", e)
        return None

    except Exception as e:
        print("DB error:", e)
        return None



#  ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Load model
    try:
        model = load_model_once()
    except Exception as e:
        flash(f"Model Load Error: {e}")
        return redirect(url_for("index"))

    # Read uploaded or webcam image
    if "image" in request.files and request.files["image"].filename:
        img = Image.open(request.files["image"].stream)

    else:
        webcam_data = request.form.get("webcam_image", "")
        if webcam_data:
            _, encoded = webcam_data.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        else:
            flash("No image provided")
            return redirect(url_for("index"))

    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]

    # Top prediction
    top_indices = preds.argsort()[::-1][:5]
    pred_idx = int(top_indices[0])
    pred_name = CLASS_NAMES[pred_idx]

    # Fetch DB details
    info = get_db_info(pred_name)

    # full top-5 list
    top5 = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices]

    return render_template(
        "result.html",
        prediction=pred_name,
        info=info,
        top5=top5
    )


@app.route("/predict_json", methods=["POST"])
def predict_json():
    model = load_model_once()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img = Image.open(request.files["image"].stream)
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]

    top = preds.argsort()[::-1][:5]
    top5 = [{"class": CLASS_NAMES[i], "prob": float(preds[i])} for i in top]

    return jsonify(top5)


@app.route("/health")
def health():
    return "ok"



#  MAIN

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

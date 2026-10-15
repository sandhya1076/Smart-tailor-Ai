from flask import Flask, request, jsonify
import os
import sys
import uuid

# ============================================================
# PROJECT PATH
# ============================================================

AI_MODEL_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

if AI_MODEL_DIR not in sys.path:
    sys.path.append(AI_MODEL_DIR)


# ============================================================
# AI PIPELINE
# ============================================================

from measurement.ai_pipeline import analyze_body


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = os.path.join(
    AI_MODEL_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Smart Tailor AI API is running",
        "endpoint": "/predict"
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    image_path = None

    try:

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "status": "error",
                "message": "No image uploaded."
            }), 400


        image = request.files["image"]


        if image.filename == "":

            return jsonify({
                "status": "error",
                "message": "Empty image filename."
            }), 400


        # ----------------------------------------------------
        # CHECK HEIGHT
        # ----------------------------------------------------

        height = request.form.get("height")


        if height is None:

            return jsonify({
                "status": "error",
                "message": "Height is required in centimeters."
            }), 400


        try:

            height = float(height)

        except ValueError:

            return jsonify({
                "status": "error",
                "message": "Height must be a number."
            }), 400


        if height <= 0 or height > 250:

            return jsonify({
                "status": "error",
                "message": "Please enter a valid height in cm."
            }), 400


        # ----------------------------------------------------
        # SAVE TEMPORARY IMAGE
        # ----------------------------------------------------

        extension = os.path.splitext(
            image.filename
        )[1].lower()


        if extension not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            return jsonify({
                "status": "error",
                "message": "Only JPG, JPEG and PNG images are supported."
            }), 400


        unique_name = (
            str(uuid.uuid4())
            + extension
        )


        image_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )


        image.save(image_path)


        # ----------------------------------------------------
        # RUN AI
        # ----------------------------------------------------

        result = analyze_body(
            image_path,
            height
        )


        # ----------------------------------------------------
        # API RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "status": "success",

            "measurements": {

                "height_cm":
                    result["height_cm"],

                "shoulder_width_cm":
                    result["shoulder_width_cm"],

                "hip_width_cm":
                    result["hip_width_cm"],

                "arm_length_cm":
                    result["arm_length_cm"],

                "leg_length_cm":
                    result["leg_length_cm"]
            },

            "size_recommendation": {

                "size":
                    result["recommended_size"],

                "type":
                    "experimental"
            }
        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message":
                "AI processing failed.",

            "details":
                str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY IMAGE
        # ----------------------------------------------------

        if (
            image_path is not None
            and os.path.exists(image_path)
        ):

            os.remove(image_path)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
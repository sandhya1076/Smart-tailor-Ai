from flask import Flask, request, jsonify
import os
import sys

# Allow imports from ai-model
sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__)
    )
)

from measurement.ai_pipeline import analyze_body


app = Flask(__name__)


@app.route("/")
def home():

    return jsonify({
        "message": "Smart Tailor AI API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400


        image = request.files["image"]


        # Get height
        height = request.form.get("height")

        if height is None:
            return jsonify({
                "error": "Height is required"
            }), 400


        height = float(height)


        # Temporary upload path
        upload_folder = "uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True
        )


        image_path = os.path.join(
            upload_folder,
            image.filename
        )


        image.save(image_path)


        # Run AI
        result = analyze_body(
            image_path,
            height
        )


        # Delete uploaded image after processing
        if os.path.exists(image_path):
            os.remove(image_path)


        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
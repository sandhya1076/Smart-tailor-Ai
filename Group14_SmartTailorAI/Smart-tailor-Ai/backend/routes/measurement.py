import os
import uuid

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from database import db
from models.database_models import User, Measurement, Prediction

measurement_bp = Blueprint("measurement", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(image_file, image_type):
    if not image_file or image_file.filename == "":
        return None

    if not allowed_file(image_file.filename):
        return None

    original_name = secure_filename(image_file.filename)
    unique_name = f"{image_type}_{uuid.uuid4().hex}_{original_name}"

    file_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        unique_name
    )

    image_file.save(file_path)

    return file_path


@measurement_bp.route("/upload", methods=["POST"])
def upload_measurement():
    user_id = request.form.get("user_id", type=int)
    height = request.form.get("height", type=float)

    front_image = request.files.get("front_image")
    side_image = request.files.get("side_image")

    if not user_id or not height:
        return jsonify({
            "message": "user_id and height are required"
        }), 400

    user = User.query.get(user_id)

    if not user:
        return jsonify({"message": "User not found"}), 404

    if not front_image or not side_image:
        return jsonify({
            "message": "front_image and side_image are required"
        }), 400

    front_image_path = save_image(front_image, "front")
    side_image_path = save_image(side_image, "side")

    if not front_image_path or not side_image_path:
        return jsonify({
            "message": "Only PNG, JPG and JPEG files are allowed"
        }), 400

    measurement = Measurement(
        user_id=user_id,
        height=height,
        front_image_path=front_image_path,
        side_image_path=side_image_path
    )

    db.session.add(measurement)
    db.session.commit()

    return jsonify({
        "message": "Images uploaded successfully",
        "measurement_id": measurement.id,
        "measurement": measurement.to_dict()
    }), 201


@measurement_bp.route("/<int:user_id>", methods=["GET"])
def get_measurement_history(user_id):
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message": "User not found"}), 404

    measurements = (
        Measurement.query
        .filter_by(user_id=user_id)
        .order_by(Measurement.created_at.desc())
        .all()
    )

    results = []

    for measurement in measurements:
        item = measurement.to_dict()

        if measurement.prediction:
            item["recommended_size"] = measurement.prediction.predicted_size
            item["confidence"] = measurement.prediction.confidence
        else:
            item["recommended_size"] = None
            item["confidence"] = None

        results.append(item)

    return jsonify(results), 200
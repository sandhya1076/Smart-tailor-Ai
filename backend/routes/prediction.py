from flask import Blueprint, request, jsonify

from database import db
from models.database_models import User, Measurement, Prediction
from services.ai_service import run_ai_prediction

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/measurement/predict", methods=["POST"])
def predict_measurement():
    data = request.get_json()

    if not data:
        return jsonify({
            "message": "Request body is required"
        }), 400

    measurement_id = data.get("measurement_id")

    if not measurement_id:
        return jsonify({
            "message": "measurement_id is required"
        }), 400

    measurement = Measurement.query.get(measurement_id)

    if not measurement:
        return jsonify({
            "message": "Measurement record not found"
        }), 404

    try:
        ai_result = run_ai_prediction(
            front_image_path=measurement.front_image_path,
            side_image_path=measurement.side_image_path,
            height=measurement.height
        )

        ai_measurements = ai_result["measurements"]

        measurement.shoulder = ai_measurements.get("shoulder")
        measurement.chest = ai_measurements.get("chest")
        measurement.waist = ai_measurements.get("waist")
        measurement.hip = ai_measurements.get("hip")
        measurement.arm = ai_measurements.get("arm")
        measurement.inseam = ai_measurements.get("inseam")

        if measurement.prediction:
            prediction = measurement.prediction
            prediction.predicted_size = ai_result["recommended_size"]
            prediction.confidence = ai_result["confidence"]
        else:
            prediction = Prediction(
                user_id=measurement.user_id,
                measurement_id=measurement.id,
                predicted_size=ai_result["recommended_size"],
                confidence=ai_result["confidence"]
            )

            db.session.add(prediction)

        db.session.commit()

        return jsonify({
            "message": "Prediction completed successfully",
            "measurements": measurement.to_dict(),
            "recommended_size": prediction.predicted_size,
            "confidence": prediction.confidence
        }), 200

    except Exception as error:
        db.session.rollback()

        return jsonify({
            "message": "Prediction failed",
            "error": str(error)
        }), 500


@prediction_bp.route("/customers", methods=["GET"])
def get_customers():
    customers = User.query.filter_by(role="customer").all()

    return jsonify([
        customer.to_dict()
        for customer in customers
    ]), 200


@prediction_bp.route("/customers/<int:customer_id>", methods=["GET"])
def get_customer_details(customer_id):
    customer = User.query.filter_by(
        id=customer_id,
        role="customer"
    ).first()

    if not customer:
        return jsonify({
            "message": "Customer not found"
        }), 404

    measurements = (
        Measurement.query
        .filter_by(user_id=customer_id)
        .order_by(Measurement.created_at.desc())
        .all()
    )

    history = []

    for measurement in measurements:
        item = measurement.to_dict()

        if measurement.prediction:
            item["recommended_size"] = (
                measurement.prediction.predicted_size
            )
            item["confidence"] = (
                measurement.prediction.confidence
            )
        else:
            item["recommended_size"] = None
            item["confidence"] = None

        history.append(item)

    return jsonify({
        "customer": customer.to_dict(),
        "measurement_history": history
    }), 200
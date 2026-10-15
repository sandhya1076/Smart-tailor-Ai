import os
from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from database import db

from routes.auth import auth_bp
from routes.measurement import measurement_bp
from routes.prediction import prediction_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)

    with app.app_context():
        from models.database_models import User, Measurement, Prediction
        db.create_all()

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(measurement_bp, url_prefix="/api/measurement")
    app.register_blueprint(prediction_bp, url_prefix="/api")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "Backend is running"
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
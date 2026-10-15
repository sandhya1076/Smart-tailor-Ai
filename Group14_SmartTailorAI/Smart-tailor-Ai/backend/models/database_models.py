from datetime import datetime
from database import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="customer")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    measurements = db.relationship(
        "Measurement",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )

    predictions = db.relationship(
        "Prediction",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role
        }


class Measurement(db.Model):
    __tablename__ = "measurements"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    height = db.Column(db.Float, nullable=True)
    shoulder = db.Column(db.Float, nullable=True)
    chest = db.Column(db.Float, nullable=True)
    waist = db.Column(db.Float, nullable=True)
    hip = db.Column(db.Float, nullable=True)
    arm = db.Column(db.Float, nullable=True)
    inseam = db.Column(db.Float, nullable=True)

    front_image_path = db.Column(db.String(255), nullable=True)
    side_image_path = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship(
        "Prediction",
        backref="measurement",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "height": self.height,
            "shoulder": self.shoulder,
            "chest": self.chest,
            "waist": self.waist,
            "hip": self.hip,
            "arm": self.arm,
            "inseam": self.inseam,
            "created_at": self.created_at.isoformat()
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    measurement_id = db.Column(
        db.Integer,
        db.ForeignKey("measurements.id"),
        nullable=False
    )

    predicted_size = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "measurement_id": self.measurement_id,
            "predicted_size": self.predicted_size,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }
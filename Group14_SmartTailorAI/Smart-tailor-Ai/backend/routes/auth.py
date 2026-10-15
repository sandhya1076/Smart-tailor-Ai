import re

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from database import db
from models.database_models import User


auth_bp = Blueprint("auth", __name__)


def is_valid_email(email):
    """
    Basic email validation.

    Examples accepted:
    user@example.com
    pratiksha.bolade@gmail.com

    Examples rejected:
    user
    user@
    @example.com
    user@example
    user example@gmail.com
    """
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return re.fullmatch(email_pattern, email) is not None


def validate_password(password):
    """
    Returns an error message if invalid.
    Returns None if valid.
    """

    if not isinstance(password, str):
        return "Password must be text"

    if not password.strip():
        return "Password cannot be empty or only spaces"

    if len(password) < 8:
        return "Password must contain at least 8 characters"

    if len(password) > 64:
        return "Password cannot contain more than 64 characters"

    return None


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "message": "Request body is required"
        }), 400

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "customer")

    # Validate name
    if not isinstance(name, str) or not name.strip():
        return jsonify({
            "message": "Name is required"
        }), 400

    name = name.strip()

    # Validate email type and format
    if not isinstance(email, str) or not email.strip():
        return jsonify({
            "message": "Email is required"
        }), 400

    email = email.strip().lower()

    if not is_valid_email(email):
        return jsonify({
            "message": "Please enter a valid email address"
        }), 400

    # Validate password
    password_error = validate_password(password)

    if password_error:
        return jsonify({
            "message": password_error
        }), 400

    # Validate role
    if role not in ["customer", "tailor"]:
        return jsonify({
            "message": "Role must be customer or tailor"
        }), 400

    # Check duplicate email
    existing_user = User.query.filter_by(email=email).first()

    if existing_user:
        return jsonify({
            "message": "Email is already registered"
        }), 409

    # Create user with hashed password
    new_user = User(
        name=name,
        email=email,
        password=generate_password_hash(password),
        role=role
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "User registered successfully",
        "user": new_user.to_dict()
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "message": "Request body is required"
        }), 400

    email = data.get("email")
    password = data.get("password")

    if not isinstance(email, str) or not email.strip():
        return jsonify({
            "message": "Email is required"
        }), 400

    if not isinstance(password, str) or not password:
        return jsonify({
            "message": "Password is required"
        }), 400

    email = email.strip().lower()

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({
            "message": "Invalid email or password"
        }), 401

    return jsonify({
        "message": "Login successful",
        "user": user.to_dict()
    }), 200
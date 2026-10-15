def run_ai_prediction(front_image_path, side_image_path, height):
    """
    Replace this temporary function after Person 1 provides:
    - YOLO pose model
    - measurement calculation function
    - size prediction model
    """

    measurements = {
        "height": height,
        "shoulder": 39.0,
        "chest": 86.0,
        "waist": 72.0,
        "hip": 92.0,
        "arm": 58.0,
        "inseam": 76.0
    }

    return {
        "measurements": measurements,
        "recommended_size": "M",
        "confidence": 0.91
    }
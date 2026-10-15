def calculate_scale_factor(height_cm, body_height_pixels):
    """
    Calculate centimetres represented by one pixel.
    """

    if body_height_pixels <= 0:
        raise ValueError("Body height in pixels must be greater than zero.")

    scale_factor = height_cm / body_height_pixels

    return scale_factor


def pixels_to_cm(pixel_measurement, scale_factor):
    """
    Convert a pixel measurement into centimetres.
    """

    return pixel_measurement * scale_factor
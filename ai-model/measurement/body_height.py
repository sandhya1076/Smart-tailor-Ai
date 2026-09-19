def calculate_body_height(points):
    """
    Calculate body height in pixels using
    the highest and lowest detected body points.
    """

    # Nose is the highest reliable point for our prototype
    top_y = points[0][1]

    # Ankles are the lowest body points
    left_ankle_y = points[15][1]
    right_ankle_y = points[16][1]

    bottom_y = max(left_ankle_y, right_ankle_y)

    body_height_pixels = bottom_y - top_y

    return body_height_pixels
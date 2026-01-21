"""
Shared geometry and drawing utilities for dataset generation.
Consolidates duplicated code from dataset generators.
"""

import numpy as np
from PIL import Image, ImageDraw


def draw_pacman(draw, center, radius, angle, mouth_width, bg_color, shape_color):
    """
    Draw a pacman (circle with missing wedge) at given center.

    Args:
        draw: PIL ImageDraw object
        center: (x, y) center coordinates
        radius: Pacman radius in pixels
        angle: Angle of mouth opening in degrees
        mouth_width: Width of mouth opening in degrees
        bg_color: Background color
        shape_color: Pacman fill color
    """
    cx, cy = center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]

    # Draw full circle
    draw.ellipse(bbox, fill=shape_color)

    # Erase wedge to form pacman mouth
    mouth_half = mouth_width / 2
    start = angle - mouth_half
    end = angle + mouth_half
    draw.pieslice(bbox, start=start, end=end, fill=bg_color)


def draw_square(ix0, iy0, inner, bg_color, shape_color, img_size):
    """
    Draw a filled square.

    Args:
        ix0, iy0: Top-left corner coordinates
        inner: Side length
        bg_color: Background color
        shape_color: Square fill color
        img_size: Canvas size

    Returns:
        PIL Image, inner, ix0, iy0
    """
    img = Image.new("L", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ix0, iy0, ix0 + inner, iy0 + inner], fill=shape_color)
    return img, inner, ix0, iy0


def draw_rectangle(ix0, iy0, inner, bg_color, shape_color, img_size):
    """
    Draw a filled rectangle (2:1 aspect ratio).

    Args:
        ix0, iy0: Top-left corner coordinates
        inner: Width
        bg_color: Background color
        shape_color: Rectangle fill color
        img_size: Canvas size

    Returns:
        PIL Image, inner, ix0, iy0
    """
    img = Image.new("L", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ix0, iy0, ix0 + inner, iy0 + int(inner / 2)], fill=shape_color)
    return img, inner, ix0, iy0


def draw_trapezium(ix0, iy0, inner, bg_color, shape_color, img_size):
    """
    Draw a filled trapezium.

    Args:
        ix0, iy0: Bottom-left corner coordinates
        inner: Base width
        bg_color: Background color
        shape_color: Trapezium fill color
        img_size: Canvas size

    Returns:
        PIL Image, inner, ix0, iy0
    """
    img = Image.new("L", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    bottom_left = (ix0, iy0)
    bottom_right = (ix0 + inner, iy0)
    top_left = (ix0 - inner // 2, iy0 + inner)
    top_right = (ix0 + inner + inner // 2, iy0 + inner)

    trapezium_points = [bottom_left, bottom_right, top_right, top_left]
    draw.polygon(trapezium_points, fill=shape_color)

    return img, inner, ix0, iy0


def draw_triangle(ix0, iy0, inner, bg_color, shape_color, img_size):
    """
    Draw a filled regular triangle.

    Args:
        ix0, iy0: Top-left bounding box corner
        inner: Circumradius
        bg_color: Background color
        shape_color: Triangle fill color
        img_size: Canvas size

    Returns:
        PIL Image, inner, ix0, iy0
    """
    img = Image.new("L", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)
    center = (ix0 + inner // 2, iy0 + inner // 2)
    draw.regular_polygon((*center, inner), n_sides=3, fill=shape_color)
    return img, inner, ix0, iy0


def draw_hexagon(ix0, iy0, inner, bg_color, shape_color, img_size):
    """
    Draw a filled regular hexagon.

    Args:
        ix0, iy0: Top-left bounding box corner
        inner: Circumradius
        bg_color: Background color
        shape_color: Hexagon fill color
        img_size: Canvas size

    Returns:
        PIL Image, inner, ix0, iy0
    """
    img = Image.new("L", (img_size, img_size), color=bg_color)
    draw = ImageDraw.Draw(img)
    center = (ix0 + inner // 2, iy0 + inner // 2)
    draw.regular_polygon((*center, inner), n_sides=6, fill=shape_color)
    return img, inner, ix0, iy0


def calculate_inward_angles(base_centers, center_x, center_y):
    """
    Calculate angles from corner positions pointing toward a center.

    Args:
        base_centers: List of (x, y) corner positions
        center_x, center_y: Target center coordinates

    Returns:
        List of angles in degrees
    """
    angles = []
    for cx, cy in base_centers:
        angle = np.degrees(np.arctan2(center_y - cy, center_x - cx))
        angles.append(angle % 360)
    return angles


def get_square_corner_positions(img_size, side_length):
    """
    Get corner positions for a square centered in the image.

    Args:
        img_size: Canvas size
        side_length: Square side length

    Returns:
        List of 4 (x, y) corner positions
    """
    center_x, center_y = img_size // 2, img_size // 2
    half_side = side_length // 2

    return [
        (center_x - half_side, center_y - half_side),  # Top-left
        (center_x + half_side, center_y - half_side),  # Top-right
        (center_x + half_side, center_y + half_side),  # Bottom-right
        (center_x - half_side, center_y + half_side)   # Bottom-left
    ]


def get_rectangle_corner_positions(img_size, width, height):
    """
    Get corner positions for a rectangle centered in the image.

    Args:
        img_size: Canvas size
        width: Rectangle width
        height: Rectangle height

    Returns:
        List of 4 (x, y) corner positions
    """
    center_x, center_y = img_size // 2, img_size // 2
    half_width = width // 2
    half_height = height // 2

    return [
        (center_x - half_width, center_y - half_height),  # Top-left
        (center_x + half_width, center_y - half_height),  # Top-right
        (center_x + half_width, center_y + half_height),  # Bottom-right
        (center_x - half_width, center_y + half_height)   # Bottom-left
    ]


def check_bounds(centers, radius, img_size, margin=1):
    """
    Check if all centers are within bounds with given margin.

    Args:
        centers: List of (x, y) positions
        radius: Object radius
        img_size: Canvas size
        margin: Minimum distance from edges

    Returns:
        True if all centers are valid, False otherwise
    """
    for cx, cy in centers:
        if not (radius + margin <= cx <= img_size - radius - margin and
                radius + margin <= cy <= img_size - radius - margin):
            return False
    return True

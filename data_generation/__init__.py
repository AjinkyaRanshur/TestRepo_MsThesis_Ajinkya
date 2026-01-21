"""
Data generation package for visual illusion datasets.
Consolidates dataset generation logic with shared utilities.
"""

from .geometry import (
    draw_pacman,
    draw_square,
    draw_rectangle,
    draw_trapezium,
    draw_triangle,
    draw_hexagon,
    calculate_inward_angles,
    get_square_corner_positions,
    get_rectangle_corner_positions,
    check_bounds,
)

__all__ = [
    "draw_pacman",
    "draw_square",
    "draw_rectangle",
    "draw_trapezium",
    "draw_triangle",
    "draw_hexagon",
    "calculate_inward_angles",
    "get_square_corner_positions",
    "get_rectangle_corner_positions",
    "check_bounds",
]

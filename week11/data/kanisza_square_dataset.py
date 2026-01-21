# -*- coding: utf-8 -*-
"""Kanizsa Square-Only Illusion Dataset Generator (32x32)"""

import numpy as np
from PIL import Image, ImageDraw
import os
import csv

IMG_SIZE = 32  # Reduced canvas size

def draw_square(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ix0, iy0, ix0+inner, iy0+inner], fill=shape_color)
    return img, inner, ix0, iy0

def draw_pacman(draw, center, radius, angle, mouth_width, bg_color, shape_color):
    """Draw a pacman (circle with missing wedge) at given center."""
    cx, cy = center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=shape_color)
    mouth_width = mouth_width / 2
    start = angle - mouth_width
    end = angle + mouth_width
    draw.pieslice(bbox, start=start, end=end, fill=bg_color)

def get_square_geometry():
    """Calculate base centers and inward-facing angles for square."""
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    
    # Square with illusory size around 10-12 pixels for 32x32 canvas
    side_length = 10
    half_side = side_length // 2
    base_centers = [
        (center_x - half_side, center_y - half_side),  # Top-left
        (center_x + half_side, center_y - half_side),  # Top-right
        (center_x + half_side, center_y + half_side),  # Bottom-right
        (center_x - half_side, center_y + half_side)   # Bottom-left
    ]
    
    # Calculate angles from each corner toward center
    all_in_angles = []
    for cx, cy in base_centers:
        angle = np.degrees(np.arctan2(center_y - cy, center_x - cx))
        all_in_angles.append(angle % 360)
    
    mouth_width = 90
    return base_centers, all_in_angles, mouth_width

def draw_inducers(dx, dy, radius, condition, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

    base_centers, all_in_angles, mouth_width = get_square_geometry()
    centers = [(x + dx, y + dy) for (x, y) in base_centers]

    # Check bounds
    MARGIN = 1
    for cx, cy in centers:
        if not (radius + MARGIN <= cx <= IMG_SIZE - radius - MARGIN and
                radius + MARGIN <= cy <= IMG_SIZE - radius - MARGIN):
            return None, None, None, None

    # Assign pacman orientations
    if condition == "all_in":
        angles = all_in_angles
    elif condition == "all_out":
        angles = [(a + 180) % 360 for a in all_in_angles]
    elif condition == "random":
        rng = np.random.default_rng()
        while True:
            angles = rng.choice(all_in_angles, size=4).tolist()
            all_out_angles = [(a + 180) % 360 for a in all_in_angles]
            if angles != all_in_angles and angles != all_out_angles:
                break
    else:
        raise ValueError("Unknown condition")

    # Draw all pacmen
    for (cx, cy), a in zip(centers, angles):
        draw_pacman(draw, (cx, cy), radius, angle=a, mouth_width=mouth_width,
                   bg_color=bg_color, shape_color=shape_color)

    return img, radius, dx, dy

def find_valid_offsets(sizes_illusory, position_step, bg, fg, required):
    for max_off in range(0, IMG_SIZE//2, 1):
        valid = 0
        for dx in range(-max_off, max_off + 1, position_step):
            for dy in range(-max_off, max_off + 1, position_step):
                result = draw_inducers(dx, dy, sizes_illusory[0], "all_in", bg, fg)
                if result[0] is not None:
                    valid += 1
        if valid >= required:
            return max_off
    return None


if __name__ == "__main__":
    outdir = "kanizsa_square_dataset"
    os.makedirs(outdir, exist_ok=True)
    
    color_map_name = {0: "Black", 128: "Grey", 255: "White"}
    color_map = {
        1: {"bg": 0, "shape": 255},  # Black → White
        2: {"bg": 0, "shape": 128},  # Black → Grey
        3: {"bg": 128, "shape": 0},  # Grey → Black
        4: {"bg": 128, "shape": 255},  # Grey → White
        5: {"bg": 255, "shape": 0},  # White → Black
        6: {"bg": 255, "shape": 128},  # White → Grey
    }
    
    # Scaled down for 32x32 canvas
    sizes = [10, 8, 6]  # Basic square sizes
    sizes_illusory = [3, 2]  # Pacman radii
    
    # Create metadata CSV
    csv_file = os.path.join(outdir, "dataset_metadata.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "Class", "Should_See", "Size", "Position X", 
            "Position Y", "Background Color", "Shape Color"
        ])
        writer.writeheader()
    
    # 1. Generate basic square images
    print("Generating basic squares...")
    POSITION_STEP = 2
    
    min_x = sizes[0] // 2 + 1
    min_y = sizes[0] // 2 + 1
    max_x = IMG_SIZE - sizes[0] - (sizes[0] // 2)
    max_y = IMG_SIZE - sizes[0] - (sizes[0] // 2)
    
    num_positions_x = len(range(min_x, max_x + 1, POSITION_STEP))
    num_positions_y = len(range(min_y, max_y + 1, POSITION_STEP))
    num_basic_per_shape = len(color_map) * len(sizes) * num_positions_x * num_positions_y
    
    print(f"Will generate {num_basic_per_shape} basic squares")
    
    # Generate basic squares
    cls_dir = os.path.join(outdir, "square")
    os.makedirs(cls_dir, exist_ok=True)
    i = 0
    
    for key, colors in color_map.items():
        bg_color, shape_color = colors["bg"], colors["shape"]
        for sz in sizes:
            for x in range(min_x, max_x + 1, POSITION_STEP):
                for y in range(min_y, max_y + 1, POSITION_STEP):
                    img, size, positionx, positiony = draw_square(
                        x, y, sz, bg_color, shape_color
                    )
                    
                    info = {
                        "filename": f"square_{i}.png",
                        "Class": "square",
                        "Should_See": "square",
                        "Size": size,
                        "Position X": x,
                        "Position Y": y,
                        "Background Color": color_map_name[bg_color],
                        "Shape Color": color_map_name[shape_color],
                    }
                    
                    filepath = os.path.join(cls_dir, info["filename"])
                    img.save(filepath)
                    
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=info.keys())
                        writer.writerow(info)
                    
                    i += 1
    
    print(f"  Generated {i} basic squares")
    
    # 2. Generate Kanizsa illusion images
    num_offset_positions_needed = num_basic_per_shape // (len(color_map) * len(sizes_illusory))
    illusion_max_offset = find_valid_offsets(sizes_illusory, POSITION_STEP, 0, 255, 
                                             num_offset_positions_needed)
    print(f"Using illusion max_offset: {illusion_max_offset}")
    
    for cls in ["all_in", "all_out", "random"]:
        print(f"\nGenerating {cls} illusions...")
        cls_dir = os.path.join(outdir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        count = 0
        
        for key, colors in color_map.items():
            bg_color, shape_color = colors["bg"], colors["shape"]
            for sz in sizes_illusory:
                for dx in range(-illusion_max_offset, illusion_max_offset + 1, POSITION_STEP):
                    for dy in range(-illusion_max_offset, illusion_max_offset + 1, POSITION_STEP):
                        result = draw_inducers(dx, dy, sz, cls, bg_color, shape_color)
                        
                        if result[0] is None:
                            continue
                        
                        img, size, positionx, positiony = result
                        
                        info = {
                            "filename": f"{cls}_{count}_square.png",
                            "Class": cls,
                            "Should_See": "square",
                            "Size": size,
                            "Position X": dx,
                            "Position Y": dy,
                            "Background Color": color_map_name[bg_color],
                            "Shape Color": color_map_name[shape_color],
                        }
                        
                        filepath = os.path.join(cls_dir, info["filename"])
                        img.save(filepath)
                        
                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=info.keys())
                            writer.writerow(info)
                        
                        count += 1
        
        print(f"  {cls}: {count} images generated")
    
    print(f"\nDataset generation complete!")
    print(f"Metadata saved to: {csv_file}")

# -*- coding: utf-8 -*-
"""Kanizsa Illusion Dataset Generator - Fixed"""

import numpy as np
from PIL import Image, ImageDraw
import os
import csv

IMG_SIZE = 128  # canvas size

def draw_square(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ix0, iy0, ix0+inner, iy0+inner], fill=shape_color)
    return img, inner, ix0, iy0

def draw_rectangle(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ix0, iy0, ix0+inner, iy0+int(inner/2)], fill=shape_color)
    return img, inner, ix0, iy0

def draw_trapezium(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Define trapezium coordinates
    bottom_left = (ix0, iy0)
    bottom_right = (ix0 + inner, iy0)
    top_left = (ix0 - inner//2, iy0 + inner)
    top_right = (ix0 + inner + inner//2, iy0 + inner)
    
    trapezium_points = [bottom_left, bottom_right, top_right, top_left]
    draw.polygon(trapezium_points, fill=shape_color)
    
    return img, inner, ix0, iy0

def draw_triangle(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.regular_polygon((ix0+inner//2, iy0+inner//2, inner), n_sides=3, fill=shape_color)
    return img, inner, ix0, iy0

def draw_hexagon(ix0, iy0, inner, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.regular_polygon((ix0+inner//2, iy0+inner//2, inner), n_sides=6, fill=shape_color)
    return img, inner, ix0, iy0

def draw_pacman(draw, center, radius, angle, mouth_width, bg_color, shape_color):
    """Draw a pacman (circle with missing wedge) at given center."""
    cx, cy = center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    # full circle
    draw.ellipse(bbox, fill=shape_color)
    mouth_width = mouth_width / 2
    # erase a wedge to form pacman mouth
    start = angle - mouth_width
    end = angle + mouth_width
    draw.pieslice(bbox, start=start, end=end, fill=bg_color)

def get_shape_geometry(shape):
    """Calculate base centers and inward-facing angles for each shape."""
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    
    if shape == "square":
        # 4 corners of a square - scaled down to match training shapes
        # Illusory square should be around 40-50 pixels
        side_length = 40
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
        
    elif shape == "rectangle":
        # 4 corners of a rectangle - make it clearly wider than tall
        # Based on your draw_rectangle: draws rectangle with width=inner, height=inner/2
        width = 60  # Make it even wider for clearer rectangle
        height = 30  # 2:1 ratio
        half_width = width // 2
        half_height = height // 2
        base_centers = [
            (center_x - half_width, center_y - half_height),  # Top-left
            (center_x + half_width, center_y - half_height),  # Top-right
            (center_x + half_width, center_y + half_height),  # Bottom-right
            (center_x - half_width, center_y + half_height)   # Bottom-left
        ]
        # Need diagonal angles pointing toward the rectangle center
        # But adjust so mouths create straight horizontal edges
        all_in_angles = [
            45,   # Top-left: southeast
            135,  # Top-right: southwest
            225,  # Bottom-right: northwest
            315   # Bottom-left: northeast
        ]
        mouth_width = 90  # Narrower mouth for straighter edges
        
        
    elif shape == "triangle":
        # 3 vertices of an equilateral triangle - scaled down
        # Triangle should be around 40-45 pixels per side
        tri_size = 40
        height = int(tri_size * np.sqrt(3) / 2)  # Height of equilateral triangle
        base_centers = [
            (center_x, center_y - int(height * 2/3)),           # Top vertex
            (center_x + tri_size//2, center_y + int(height/3)), # Bottom-right vertex
            (center_x - tri_size//2, center_y + int(height/3))  # Bottom-left vertex
        ]
        # Calculate angles from each vertex toward center
        all_in_angles = []
        for cx, cy in base_centers:
            angle = np.degrees(np.arctan2(center_y - cy, center_x - cx))
            all_in_angles.append(angle % 360)
        mouth_width = 60  # Reduced from 120 for better triangle illusion
        
    elif shape == "trapezium":
        # 4 corners of a trapezoid - scaled down and properly aligned
        # Trapezium: narrower at top, wider at bottom
        top_width = 30
        bottom_width = 60
        height = 40
        base_centers = [
            (center_x - top_width//2, center_y - height//2),     # Top-left
            (center_x + top_width//2, center_y - height//2),     # Top-right
            (center_x + bottom_width//2, center_y + height//2),  # Bottom-right
            (center_x - bottom_width//2, center_y + height//2)   # Bottom-left
        ]
        # Calculate angles from each corner toward center
        all_in_angles = []
        for cx, cy in base_centers:
            angle = np.degrees(np.arctan2(center_y - cy, center_x - cx))
            all_in_angles.append(angle % 360)
        mouth_width = 100
        
    elif shape == "hexagon":
        # 6 vertices of a regular hexagon - scaled down
        # Hexagon should be around 35-40 pixels radius
        hex_radius = 30
        base_centers = []
        # Generate vertices at 60-degree intervals, starting from top
        for i in range(6):
            angle = i * 60  # Start at 90 degrees (top) and go clockwise
            angle_rad = np.deg2rad(angle)
            x = center_x + int(hex_radius * np.cos(angle_rad))
            y = center_y - int(hex_radius * np.sin(angle_rad))
            base_centers.append((x, y))
        
        # Calculate angles from each vertex toward center
        all_in_angles = []
        for cx, cy in base_centers:
            angle = np.degrees(np.arctan2(center_y - cy, center_x - cx))
            all_in_angles.append(angle % 360)
        mouth_width = 120
    
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    return base_centers, all_in_angles, mouth_width

def draw_inducers(dx, dy, radius, condition, shape, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Get geometry for this shape
    base_centers, all_in_angles, mouth_width = get_shape_geometry(shape)
    
    # Apply offset
    centers = [(x + dx, y + dy) for (x, y) in base_centers]

    # Check bounds
    MARGIN = 1
    for cx, cy in centers:
        if not (radius + MARGIN <= cx <= IMG_SIZE - radius - MARGIN and
                radius + MARGIN <= cy <= IMG_SIZE - radius - MARGIN):
            return None, None, None, None

    # Assign pacman orientations based on condition
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
    
    # Compute shape-centered Y reference (once)
    center_y = np.mean([cy for _, cy in centers])

    # Draw all pacmen
    for (cx,cy), a in zip(centers, angles):
        if shape == "trapezium":
            if cy < center_y:
                mouth_width=100
            else:
                mouth_width=60
        draw_pacman(
            draw,
            (cx,cy),
            radius,
            angle=a,
            mouth_width=mouth_width,
            bg_color=bg_color,
            shape_color=shape_color)

    return img, radius, dx, dy

def find_valid_offsets(shape, sizes_illusory, position_step, bg, fg):
    # Try max_offset values until you reach the required number of offsets
    required = num_basic_per_shape // (len(color_map) * len(sizes_illusory))
    
    for max_off in range(0, IMG_SIZE//2, 1):   # try 0,1,2,... until hit requirement
        valid = 0
        
        for dx in range(-max_off, max_off + 1, position_step):
            for dy in range(-max_off, max_off + 1, position_step):
                # count offsets that keep all pacmen in bounds
                result = draw_inducers(dx, dy, sizes_illusory[0], "all_in",
                                       shape, bg, fg)
                if result[0] is not None:
                    valid += 1
        
        if valid >= required:
            return max_off
    
    return None


if __name__ == "__main__":
    outdir = "visual_illusion_dataset"
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
    
    sizes = [40, 30, 20]
    sizes_illusory = [10, 7, 5]
    shape_list = ["square", "rectangle", "trapezium", "triangle", "hexagon"]
    
    # Shape drawing function mapping
    shape_functions = {
        "square": draw_square,
        "rectangle": draw_rectangle,
        "trapezium": draw_trapezium,
        "triangle": draw_triangle,
        "hexagon": draw_hexagon,
    }
    
    # Create metadata CSV
    csv_file = os.path.join(outdir, "dataset_metadata.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "Class", "Should_See", "Size", "Position X", 
            "Position Y", "Background Color", "Shape Color"
        ])
        writer.writeheader()
    
    # 1. Generate basic shape images
    print("Generating basic shapes...")
    POSITION_STEP = 3  # Sample positions every 5 pixels instead of every pixel

    # Calculate the minimum max_offset across all shapes and sizes
    # This ensures all shapes generate the same number of images

    # Calculate position range for basic shapes
    min_x = sizes[0] // 2 + 1  # Use largest size for most restrictive bounds
    min_y = sizes[0] // 2 + 1
    max_x = IMG_SIZE - sizes[0] - (sizes[0] // 2)
    max_y = IMG_SIZE - sizes[0] - (sizes[0] // 2)

    num_positions_x = len(range(min_x, max_x + 1, POSITION_STEP))
    num_positions_y = len(range(min_y, max_y + 1, POSITION_STEP))

    num_basic_per_shape = len(color_map) * len(sizes) * num_positions_x * num_positions_y
    total_basic_shapes = num_basic_per_shape * len(shape_list)

    print(f"Total basic shapes that will be generated: {total_basic_shapes}")
    print(f"Per shape: {num_basic_per_shape}")


    # Now calculate the offset needed for illusions to match this count
    target_per_shape = num_basic_per_shape  # Each illusion shape should have same count as basic

    # We have: len(color_map) * len(sizes_illusory) * num_offset_positions = target_per_shape
    # So: num_offset_positions = target_per_shape / (len(color_map) * len(sizes_illusory))
    num_offset_positions_needed = target_per_shape // (len(color_map) * len(sizes_illusory))

    # For a square grid: num_positions = (2*max_offset/POSITION_STEP + 1)^2
    # Solve for max_offset
    positions_per_axis = int(np.sqrt(num_offset_positions_needed))
    # illusion_max_offset = (positions_per_axis - 1) * POSITION_STEP // 2

    # print(f"Using illusion max_offset: {illusion_max_offset} to match basic shapes count")

    illusion_offsets = {}

    for shape in shape_list:
        # Use any color because bounds don't depend on color
        sample_bg = 0
        sample_fg = 255
        
        offset = find_valid_offsets(shape, sizes_illusory, POSITION_STEP,
                                    sample_bg, sample_fg)
        illusion_offsets[shape] = offset
        print(f"{shape} max offset = {offset}")


    
    for shape in shape_list:
        cls_dir = os.path.join(outdir, shape)
        os.makedirs(cls_dir, exist_ok=True)
        i = 0
        
        for key, colors in color_map.items():
            bg_color, shape_color = colors["bg"], colors["shape"]
            for sz in sizes:
                
                for x in range(min_x, max_x + 1, POSITION_STEP):
                    for y in range(min_y,max_y + 1, POSITION_STEP):
                        img, size, positionx, positiony = shape_functions[shape](
                            x, y, sz, bg_color, shape_color
                        )
                        
                        info = {
                            "filename": f"{shape}_{i}.png",
                            "Class": shape,
                            "Should_See": shape,
                            "Size": size,
                            "Position X": x,
                            "Position Y": y,
                            "Background Color": color_map_name[bg_color],
                            "Shape Color": color_map_name[shape_color],
                        }
                        
                        filepath = os.path.join(cls_dir, info["filename"])
                        img.save(filepath)
                        
                        # Append to CSV
                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=info.keys())
                            writer.writerow(info)
                        
                        i += 1
        print(f"  {shape}: {i} images generated")
    
    # 2. Generate Kanizsa illusion images (all_in, all_out, random)
    cls_list = ["all_in", "all_out", "random"]

    
    for cls in cls_list:
        print(f"\nGenerating {cls} illusions...")
        cls_dir = os.path.join(outdir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        count = 0
        # Track count per shape to ensure balance
        shape_counts = {shape: 0 for shape in shape_list}
        
        for shape in shape_list:

            for key, colors in color_map.items():
                bg_color, shape_color = colors["bg"], colors["shape"]
                for sz in sizes_illusory:

                    max_offset = illusion_offsets[shape]

                    for dx in range(-max_offset, max_offset + 1, POSITION_STEP):
                        for dy in range(-max_offset, max_offset + 1, POSITION_STEP):
                            result = draw_inducers(
                                dx, dy, sz, cls, shape, bg_color, shape_color
                            )
                            
                            if result[0] is None:  # Skip if out of bounds
                                continue
                            
                            img, size, positionx, positiony = result
                            
                            info = {
                                "filename": f"{cls}_{count}_{shape}.png",
                                "Class": cls,
                                "Should_See": shape,
                                "Size": size,
                                "Position X": dx,
                                "Position Y": dy,
                                "Background Color": color_map_name[bg_color],
                                "Shape Color": color_map_name[shape_color],
                            }
                            
                            filepath = os.path.join(cls_dir, info["filename"])
                            img.save(filepath)
                            
                            # Append to CSV
                            with open(csv_file, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=info.keys())
                                writer.writerow(info)
                            
                            count += 1
                            shape_counts[shape] += 1
        
        # print(f"  {cls}: {count} images generated")
        # Print counts per shape for this class
        print(f"  {cls} totals by shape:")
        for shape in shape_list:
            print(f"    {shape}: {shape_counts[shape]} images")
        print(f"  {cls} total: {sum(shape_counts.values())} images")
    
    print(f"\nDataset generation complete!")
    print(f"Metadata saved to: {csv_file}")


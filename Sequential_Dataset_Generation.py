import numpy as np
from PIL import Image, ImageDraw
import os
import csv

IMG_SIZE = 32  # canvas size


def draw_square(ix0, iy0, inner, bg_color, shape_color):
    """Draw a simple filled square in the center."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE),
                    color=bg_color)  # light gray background
    draw = ImageDraw.Draw(img)
    outer = 32
    # outer dark square
    x0 = (IMG_SIZE - outer) // 2
    y0 = (IMG_SIZE - outer) // 2
    draw.rectangle([x0, y0, x0 + outer, y0 + outer], fill=bg_color)

    draw.rectangle([ix0, iy0, ix0 + inner, iy0 + inner], fill=shape_color)

    return img, inner, ix0, iy0


def draw_pacman(draw, center, radius, angle, bg_color, shape_color):
    """Draw a pacman (circle with missing wedge) at given center."""
    cx, cy = center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    # full circle
    draw.ellipse(bbox, fill=shape_color)
    # erase a wedge to form pacman mouth
    start = angle - 45
    end = angle + 45
    draw.pieslice(bbox, start=start, end=end, fill=bg_color)


def draw_inducers(dx, dy, radius, condition, bg_color, shape_color):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

    base_centers = [(10, 10), (IMG_SIZE - 10, 10),
                    (10, IMG_SIZE - 10), (IMG_SIZE - 10, IMG_SIZE - 10)]

    centers = [(x + dx, y + dy) for (x, y) in base_centers]

    MARGIN = 1  # keep pacmen well inside the canvas
    for cx, cy in centers:
        if not (radius + MARGIN <= cx <= IMG_SIZE - radius - MARGIN and
                radius + MARGIN <= cy <= IMG_SIZE - radius - MARGIN):
            return None, None, None, None

    # assign pacman orientations as before
    all_in_angles = [45, 135, 315, 225]
    if condition == "All-in":
        angles = all_in_angles
    elif condition == "All-out":
        angles = [(a + 180) % 360 for a in all_in_angles]
    elif condition == "Random":
        rng = np.random.default_rng()
        while True:
            angles = rng.choice([45, 135, 315, 225], size=4).tolist()
            all_out_angles = [(a + 180) % 360 for a in all_in_angles]
            if angles != all_in_angles and angles != all_out_angles:
                break
    else:
        raise ValueError("Unknown condition")

    for c, a in zip(centers, angles):
        draw_pacman(
            draw,
            c,
            radius,
            angle=a,
            bg_color=bg_color,
            shape_color=shape_color)

    return img, radius, dx, dy


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

    classes = ["Square", "All-in", "All-out", "Random"]
    sizes = [11, 8, 6]
    metadata = []

    images_per_class = {cls: 0 for cls in classes}

    for cls in classes:
        cls_dir = os.path.join(outdir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        i = 0
        for key, colors in color_map.items():
            bg_color, shape_color = colors["bg"], colors["shape"]
            if cls == "Square":
                for sz in sizes:
                    # ensure square fully inside
                    max_coord = IMG_SIZE - sizes[0]
                    for x in range(0, max_coord + 1, 2):
                        for y in range(0, max_coord + 1,2):
                            i += 1
                            img, size, positionx, positiony = draw_square(
                                x, y, sz, bg_color, shape_color)
                            info = {
                                "filename": f"{cls}_{i}.png",
                                "Class": "Square",
                                "Size": size,
                                "Position X": x,
                                "Position Y": y,
                                "Background Color": color_map_name[bg_color],
                                "Shape Color": color_map_name[shape_color],
                            }
                            filepath = os.path.join(cls_dir, info["filename"])
                            img.save(filepath)
                            metadata.append(info)
                            images_per_class[cls] += 1
            else:
                pacmansize = [3, 2]
                for radius in pacmansize:
                    # define safe dx, dy ranges explicitly
                    # pacman centers start at 10 or 22
                    #You can either keep radius or the maximum radius in the shifts maximum leads to more uniformity
                    min_shift = -(10 - pacmansize[0])
                    max_shift = (IMG_SIZE - 10) - pacmansize[0] - 10
                    for dx in range(min_shift, max_shift + 1):
                        for dy in range(min_shift, max_shift + 1):
                            img, size, posx, posy = draw_inducers(
                                dx, dy, radius, cls, bg_color, shape_color)
                            if img is not None:
                                i += 1
                                info = {
                                    "filename": f"{cls}_{i}.png",
                                    "Class": cls,
                                    "Size": size,
                                    "Position X": dx,
                                    "Position Y": dy,
                                    "Background Color": color_map_name[bg_color],
                                    "Shape Color": color_map_name[shape_color],
                                }
                                filepath = os.path.join(
                                    cls_dir, info["filename"])
                                img.save(filepath)
                                metadata.append(info)
                                images_per_class[cls] += 1
                            else:
                                continue

    # save metadata
    with open(os.path.join(outdir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    # print number of images per class
    print("Images created per class:")
    for cls, count in images_per_class.items():
        print(f"{cls}: {count}")

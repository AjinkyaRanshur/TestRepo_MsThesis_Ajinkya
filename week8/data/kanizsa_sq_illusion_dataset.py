import numpy as np
from PIL import Image, ImageDraw
import os
import csv 

IMG_SIZE = 32  # canvas size

# ------------------- BASIC SHAPES -------------------


def draw_square(bg_color, shape_color):
    """Draw a simple filled square in the center."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE),
                    color=bg_color)  # light gray background
    draw = ImageDraw.Draw(img)
    outer = 32
    # Inner square size: small, normal, big
    rng = np.random.default_rng()
    inner = int(rng.choice([6, 12, 18]))
    # outer dark square
    x0 = (IMG_SIZE - outer) // 2
    y0 = (IMG_SIZE - outer) // 2
    draw.rectangle([x0, y0, x0 + outer, y0 + outer], fill=shape_color)

    ix0 = np.random.randint(x0, x0 + outer - inner + 1)
    iy0 = np.random.randint(y0, y0 + outer - inner + 1)
    draw.rectangle([ix0, iy0, ix0 + inner, iy0 + inner], fill=bg_color)
    return img,inner,ix0,iy0


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


def draw_inducers(condition, bg_color, shape_color):
    """Draw 4 pacman inducers in corners with different orientations."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)
    # base centers (same relative arrangement as before)
    base_centers = [(10, 10), (IMG_SIZE - 10, 10),
                    (10, IMG_SIZE - 10), (IMG_SIZE - 10, IMG_SIZE - 10)]
    rng = np.random.default_rng()
    radius = int(rng.choice([1, 5, 9]))
    rng = np.random.default_rng()

    # compute allowed dx range so every translated center stays within
    # [radius, IMG_SIZE-radius]
    xs = [c[0] for c in base_centers]
    ys = [c[1] for c in base_centers]
    dx_min = max(radius - x for x in xs)
    dx_max = min((IMG_SIZE - radius) - x for x in xs)
    dy_min = max(radius - y for y in ys)
    dy_max = min((IMG_SIZE - radius) - y for y in ys)

    # if something odd happens, fall back to no shift
    if dx_min > dx_max or dy_min > dy_max:
        dx, dy = 0, 0
    else:
        dx = int(rng.integers(dx_min, dx_max + 1))
        dy = int(rng.integers(dy_min, dy_max + 1))

    # apply same translation to all centers
    centers = [(x + dx, y + dy) for (x, y) in base_centers]
    all_in_angles = [45, 135, 315, 225]
    if condition == "All-in":
        angles = all_in_angles
    elif condition == "All-out":
        angles = [(a + 180) % 360 for a in all_in_angles]
    elif condition == "Random":
        rng = np.random.default_rng()
        angles = rng.choice([0, 90, 180, 270], size=4)
    else:
        raise ValueError("Unknown condition")
    for c, a in zip(centers, angles):
        draw_pacman(draw, c, radius=4, angle=a, bg_color=bg_color, shape_color=shape_color)
    return img,radius,dx,dy


# ------------------- DEMO -------------------
if __name__ == "__main__":
    num_per_class=2500
    outdir="visual_illusion_dataset"
    os.makedirs(outdir, exist_ok=True)
    color_map_name={0:"Black",128:"Grey",255:"White"}
    color_map = {
    1: {"bg": 0,   "shape": 255},   # Black → White
    2: {"bg": 0,   "shape": 128},   # Black → Grey
    3: {"bg": 128, "shape": 0},     # Grey → Black
    4: {"bg": 128, "shape": 255},   # Grey → White
    5: {"bg": 255, "shape": 0},     # White → Black
    6: {"bg": 255, "shape": 128},   # White → Grey
	}
	
	
    
    classes = ["Square", "All-in", "All-out", "Random"]
    metadata = []
    for cls in classes:
        cls_dir = os.path.join(outdir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(num_per_class):
            scheme_id = np.random.randint(1, 7)
            colors = color_map[scheme_id]
            bg_color, shape_color = colors["bg"], colors["shape"]
            
            if cls == "Square":
                img,size,positionx,positiony = draw_square(bg_color, shape_color)
                # collect metadata
                info = {
                    "filename": f"{cls}_{i}.png",
                    "Class": "Square",
                    "Size":size,
                    "Position X":positionx,
                    "Position Y":positiony,
                    "Background Color":color_map_name[bg_color],
                    "Shape Color":color_map_name[shape_color],
                    # inner square size & position recorded
                }
            else:
                img,size,positionx,positiony = draw_inducers(cls,bg_color, shape_color)
                info = {
                    "filename": f"{cls}_{i}.png",
                    "Class": cls,
                    "Size":size,
                    "Position X":positionx,
                    "Position Y":positiony,
                    "Background Color":color_map_name[bg_color],
                    "Shape Color":color_map_name[shape_color],
                    # radius, dx, dy could also be stored if you return them
                }

            filepath = os.path.join(cls_dir, info["filename"])
            img.save(filepath)
            metadata.append(info)

    # save metadata
    with open(os.path.join(outdir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

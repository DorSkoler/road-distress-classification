import os
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means
from skimage.measure import shannon_entropy
from itertools import product
import json

# ─── Metrics ────────────────────────────────────────────────────────────────
def sharpness_metric(img):
    """Variance of Laplacian: higher = crisper edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def entropy_metric(img):
    """Shannon entropy on grayscale: higher = more detail & contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

def quality_score(img):
    """Combine metrics: you can re-weight as you like."""
    return entropy_metric(img) + 0.01 * sharpness_metric(img)


# ─── 1) DENOISING OPTIMIZER ─────────────────────────────────────────────────
def optimize_denoise(img):
    """Non-local means denoising: search over handful of settings."""
    img_f = img_as_float(img)
    best, best_score = img, quality_score(img)
    # grid of parameters
    param_grid = {
      "h":      [0.8, 1.0, 1.2],
      "patch_size":    [5, 7],
      "patch_distance":[3, 5]
    }
    for h, ps, pd in product(param_grid["h"],
                             param_grid["patch_size"],
                             param_grid["patch_distance"]):
        den = denoise_nl_means(img_f,
                               h=h,
                               patch_size=ps,
                               patch_distance=pd,
                               channel_axis=-1)
        den_u8 = img_as_ubyte(den)
        s = quality_score(den_u8)
        if s > best_score:
            best_score, best = s, den_u8
    print(f"[Denoise] best score = {best_score:.2f}")
    return best


# ─── 2) CLAHE OPTIMIZER ──────────────────────────────────────────────────────
def optimize_clahe(img):
    """Apply CLAHE on L-channel in LAB space, pick best clipLimit/tileGrid."""
    best, best_score = img, quality_score(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    for clip, grid in product([1.0, 2.0, 3.0],
                              [(8,8), (16,16), (32,32)]):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        s = quality_score(out)
        if s > best_score:
            best_score, best = s, out
    print(f"[CLAHE] best score = {best_score:.2f}")
    return best


# ─── 3) SHARPEN OPTIMIZER ────────────────────────────────────────────────────
def optimize_sharpen(img):
    """Unsharp‐mask style sharpening: blend image + negative blur."""
    best, best_score = img, quality_score(img)
    for alpha in [1.2, 1.5, 1.8]:   # proportion of original
        beta = 1.0 - alpha          # proportion of blurred 
        blurred = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
        sharp = cv2.addWeighted(img, alpha, blurred, beta, 0)
        s = quality_score(sharp)
        if s > best_score:
            best_score, best = s, sharp
    print(f"[Sharpen] best score = {best_score:.2f}")
    return best


# ─── ANNOTATION OVERLAY ─────────────────────────────────────────────────────-
def overlay_annotations(img, json_path):
    if not os.path.exists(json_path):
        return img
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tags = data.get('tags', [])
        y0, dy = 30, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 255, 0)
        thickness = 2
        img_annot = img.copy()
        for i, tag in enumerate(tags):
            text = str(tag.get('value', ''))
            y = y0 + i * dy
            cv2.putText(img_annot, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return img_annot
    except Exception as e:
        print(f"[WARN] Could not overlay annotation from {json_path}: {e}")
        return img


# ─── PIPELINE FOR ONE IMAGE ──────────────────────────────────────────────────
def enhance_pipeline(img):
    den = optimize_denoise(img)
    cla = optimize_clahe(den)
    shp = optimize_sharpen(cla)
    return shp


# ─── BATCH PROCESSING ────────────────────────────────────────────────────────
def batch_enhance(input_root, output_root, side_by_side=False, draw_annotations=False):
    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.lower().endswith('.png'):
                in_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(in_path, input_root)
                out_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img = cv2.imread(in_path)
                if img is None:
                    print(f"[ERROR] Failed to read {in_path}")
                    continue
                print(f"Processing {in_path} ...")
                # Overlay annotation if requested
                orig_img = img
                if draw_annotations:
                    # Find annotation in sibling 'ann' folder
                    img_dir = os.path.dirname(in_path)
                    parent_dir = os.path.dirname(img_dir)
                    ann_dir = os.path.join(parent_dir, 'ann')
                    json_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.json')
                    orig_img = overlay_annotations(img, json_path)
                enhanced = enhance_pipeline(img)
                if side_by_side:
                    # Concatenate original and enhanced side by side
                    output_img = np.concatenate((orig_img, enhanced), axis=1)
                else:
                    output_img = enhanced
                cv2.imwrite(out_path, output_img)
                print(f"Saved image to: {out_path}")


if __name__ == "__main__":
    import sys
    side_by_side = False
    draw_annotations = False
    if '--side_by_side' in sys.argv:
        side_by_side = True
        sys.argv.remove('--side_by_side')
    if '--draw_annotations' in sys.argv:
        draw_annotations = True
        sys.argv.remove('--draw_annotations')
    if len(sys.argv) != 3:
        print("Usage: python enhance_images.py <input_root> <output_root> [--side_by_side] [--draw_annotations]")
    else:
        batch_enhance(sys.argv[1], sys.argv[2], side_by_side=side_by_side, draw_annotations=draw_annotations) 
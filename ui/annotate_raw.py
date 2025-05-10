import os
import cv2
import numpy as np
import random
import argparse
import sys
# Ensure project root is on path to import preprocessing package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.isolate_road import (
    load_model, preprocess_image, segment_road,
    refine_mask, crop_to_mask, overlay_mask, VALID_EXTENSIONS
)

# Globals for drawing polygon
current_pts = []
drawing = False

def draw_polygon(event, x, y, flags, param):
    global drawing, current_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_pts = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_pts.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_pts.append((x, y))

def show_preview(image, mask, window_name='Preview'):
    """Show preview of saved annotation and wait for approval"""
    preview = overlay_mask(image, mask)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    print("\nPreview saved annotation:")
    print("Press 'A' to approve and continue")
    print("Press 'R' to redo annotation")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('r'):
            cv2.destroyWindow(window_name)
            return False

def main(data_dir, save_dir, model_type, checkpoint, device):
    os.makedirs(save_dir, exist_ok=True)
    # Handle potential wrapper folder (e.g., data/coryell)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) == 1 and not os.path.isdir(os.path.join(data_dir, subdirs[0], 'img')):
        print(f"Descending into wrapper folder '{subdirs[0]}'")
        data_dir = os.path.join(data_dir, subdirs[0])
    # Build list of raw images (road folder/img/*.png)
    img_list = []
    for road in os.listdir(data_dir):
        img_folder = os.path.join(data_dir, road, 'img')
        if not os.path.isdir(img_folder):
            continue
        for fn in os.listdir(img_folder):
            if os.path.splitext(fn)[1].lower() in VALID_EXTENSIONS:
                img_list.append((road, os.path.join(img_folder, fn)))
    if not img_list:
        print("No images found in data directory.")
        return
    random.shuffle(img_list)
    # Load segmentation model or classical selector
    model = load_model(model_type, checkpoint, device)
    # Iterate randomly through images
    for road, img_path in img_list:
        print(f"Sample image: {road} / {os.path.basename(img_path)}")
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        # Preprocess and segment
        img_pre = preprocess_image(orig)
        mask = segment_road(model, img_pre, device)
        mask = refine_mask(mask)
        # Crop around mask
        crop_res = crop_to_mask(orig, mask, padding=20)
        if crop_res is None:
            print("No road mask detected, skipping.")
            continue
        crop_img, crop_mask = crop_res
        overlay = overlay_mask(crop_img, crop_mask)
        
        while True:  # Loop for redoing annotation if needed
            # Show overlay and annotate
            cv2.namedWindow('annotator', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('annotator', draw_polygon)
            current_pts.clear()  # Clear previous points
            
            while True:
                disp = overlay.copy()
                if current_pts:
                    pts = np.array(current_pts, np.int32).reshape((-1,1,2))
                    cv2.polylines(disp, [pts], False, (0,0,255), 2)
                cv2.imshow('annotator', disp)
                key = cv2.waitKey(20) & 0xFF
                if key == ord('s') and current_pts:
                    # Save crop and annotated mask
                    annotated = np.zeros_like(crop_mask)
                    pts_arr = np.array(current_pts, np.int32).reshape((-1,1,2))
                    cv2.fillPoly(annotated, [pts_arr], 255)
                    base = f"{road}_{os.path.splitext(os.path.basename(img_path))[0]}"
                    img_save = os.path.join(save_dir, f"{base}_crop.png")
                    mask_save = os.path.join(save_dir, f"{base}_annotated.png")
                    cv2.imwrite(img_save, crop_img)
                    cv2.imwrite(mask_save, annotated)
                    print(f"Saved crop image: {img_save}")
                    print(f"Saved annot mask:  {mask_save}")
                    
                    # Show preview and wait for approval
                    if show_preview(crop_img, annotated):
                        print("Annotation approved!")
                        cv2.destroyWindow('annotator')
                        break  # Break out of annotation loop
                    else:
                        print("Redoing annotation...")
                        continue  # Continue to redo annotation
                        
                elif key == ord('n'):
                    # Next image
                    current_pts.clear()
                    cv2.destroyWindow('annotator')
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Quitting annotation.")
                    return
            break  # Break out of redo loop if we're done with this image
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Random raw image annotator for roads")
    parser.add_argument('--data-dir', type=str, default='data', help='Root data folder (with subfolders)/')
    parser.add_argument('--save-dir', type=str, default=os.path.join('preprocessing','annotations'), help='Where to save cropped images and masks')
    parser.add_argument('--model', type=str, default='unet', help='Segmentation model type (unet, deeplabv3, deeplabv3plus, classical, segformer, advanced_classical)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for deep models')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.model, args.checkpoint, args.device) 
import os
import cv2
import numpy as np
import argparse

# Parse CLI arguments for data and save directories
parser = argparse.ArgumentParser(description="Annotation UI for road overlays")
parser.add_argument('--data-dir', type=str, default=os.path.join(os.getcwd(), 'preprocessing', 'output_adv'), help='Directory with overlay images')
parser.add_argument('--save-dir', type=str, default=os.path.join(os.getcwd(), 'preprocessing', 'annotations'), help='Directory to save annotations')
args = parser.parse_args()

# Directory containing crop and overlay files
DATA_DIR = args.data_dir
# Directory to save manual annotations
SAVE_DIR = args.save_dir

# Globals for drawing
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


def annotate():
    os.makedirs(SAVE_DIR, exist_ok=True)
    # List all crop overlay images (exclude full overlays)
    overlay_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_overlay.png') and not f.endswith('_full_overlay.png')]
    overlay_files.sort()

    for fname in overlay_files:
        base = fname.replace('_overlay.png', '')
        overlay_path = os.path.join(DATA_DIR, fname)
        mask_path = os.path.join(DATA_DIR, f'{base}_mask.png')
        overlay = cv2.imread(overlay_path)
        # Load existing crop mask or initialize blank for annotation
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(overlay.shape[:2], dtype=np.uint8)

        print(f"Annotating: {base}")
        cv2.namedWindow('annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('annotator', draw_polygon)
        while True:
            display = overlay.copy()
            # Draw current polygon
            if current_pts:
                pts = np.array(current_pts, np.int32).reshape((-1,1,2))
                cv2.polylines(display, [pts], False, (0,0,255), 2)
            cv2.imshow('annotator', display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s') and current_pts:
                # Save annotated crop mask
                annotated = mask.copy()
                pts = np.array(current_pts, np.int32).reshape((-1,1,2))
                cv2.fillPoly(annotated, [pts], 255)
                save_path = os.path.join(SAVE_DIR, f'{base}_annotated.png')
                cv2.imwrite(save_path, annotated)
                print(f"Saved annotated mask: {save_path}")
            elif key == ord('n'):
                # Next image
                current_pts.clear()
                cv2.destroyWindow('annotator')
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print("Quitting annotation.")
                return
        cv2.destroyAllWindows()

if __name__ == '__main__':
    annotate() 
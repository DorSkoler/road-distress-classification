import os
import cv2
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F

# Supported image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# Supported model types
VALID_MODEL_TYPES = {'unet', 'deeplabv3', 'deeplabv3plus', 'classical', 'segformer', 'advanced_classical'}

# Utility: pad image so height & width are multiples of a divisor, return original size
def pad_to_multiple(image, divisor=16):
    h, w = image.shape[:2]
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    pad_bottom = new_h - h
    pad_right = new_w - w
    padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    return padded, h, w

def load_model(model_type='unet', checkpoint_path=None, device='cuda'):
    """Load a U-Net model for road segmentation."""
    if model_type == 'advanced_classical':
        print("Using advanced classical polygon-based segmentation.")
        return 'advanced_classical'
    if model_type == 'classical':
        print("Using classical segmentation (no deep model).")
        return None
    if model_type == 'segformer':
        print("Loading SegFormer model for road segmentation...")
        feat_extractor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
        seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )
        seg_model.to(device)
        seg_model.eval()
        return (feat_extractor, seg_model)
    # Instantiate other deep models
    if model_type == 'unet':
        model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation=None)
    elif model_type == 'deeplabv3':
        model = smp.DeepLabV3('resnet101', encoder_weights='imagenet', classes=1, activation=None)
    elif model_type == 'deeplabv3plus':
        model = smp.DeepLabV3Plus('resnet101', encoder_weights='imagenet', classes=1, activation=None)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    # Load checkpoint if provided
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: checkpoint '{checkpoint_path}' not found. Using ImageNet-pretrained encoder.")
    model.to(device)
    model.eval()
    return model


def classical_segment_road(image):
    """Segment the road region using classical CV thresholding (no deep model)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return mask


def get_road_polygon(image, simplify_eps=5.0, margin_percent=0.1):
    # Pre-filter to reduce noise but keep edges
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    # Color thresholds
    mask_paved = cv2.inRange(hsv, (0, 0, 120), (180, 60, 255))
    mask_soil  = cv2.inRange(hsv, (5, 50, 50), (30, 255, 200))
    mask_grass = cv2.inRange(hsv, (25, 30, 30), (85, 255, 255))
    color_mask = cv2.bitwise_or(mask_paved, mask_soil)
    color_mask = cv2.bitwise_or(color_mask, mask_grass)
    # Texture: Sobel gradient magnitude
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(grad_mag, 30, 255, cv2.THRESH_BINARY)
    # Combine color + texture
    combined = cv2.bitwise_or(color_mask, texture_mask)
    # Exclude bottom-left watermark
    h, w = combined.shape
    mw, mh = int(w * margin_percent), int(h * margin_percent)
    mask_ignore = np.ones_like(combined) * 255
    mask_ignore[h-mh:, :mw] = 0
    combined = cv2.bitwise_and(combined, mask_ignore)
    # Morphological operations
    k_size = max(3, int(min(w, h) * 0.005))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Fill holes
    contours, _ = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = clean.copy()
    # Fill each contour completely
    for cnt in contours:
        cv2.drawContours(mask_filled, [cnt], -1, 255, -1)
    # Extract largest external contour
    ex_contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ex_contours:
        return np.empty((0, 2), np.int32), mask_filled
    min_area = w * h * 0.002
    ex_contours = [c for c in ex_contours if cv2.contourArea(c) > min_area]
    if not ex_contours:
        return np.empty((0, 2), np.int32), mask_filled
    main_contour = max(ex_contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(main_contour, simplify_eps, True)
    poly = approx.reshape(-1, 2)
    final_mask = np.zeros_like(combined)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    return poly, final_mask


def segment_road(model, image, device='cuda'):
    """Segment the road region using the deep model or classical fallback if no model."""
    # Advanced classical polygon-based mask
    if model == 'advanced_classical':
        _, mask = get_road_polygon(image)
        return mask
    # SegFormer inference
    if isinstance(model, tuple):
        feat_extractor, seg_model = model
        # Prepare inputs
        encoding = feat_extractor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].to(device)
        with torch.no_grad():
            outputs = seg_model(pixel_values=pixel_values)
        logits = outputs.logits  # shape [1, num_classes, H_feat, W_feat]
        # Upsample to original resolution
        upsampled = F.interpolate(logits, size=image.shape[:2], mode='bilinear', align_corners=False)
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        # Cityscapes class 1 == road
        mask = (pred == 1).astype(np.uint8) * 255
        return mask
    if model is None:
        return classical_segment_road(image)
    # Pad to meet model input size requirements
    padded_img, orig_h, orig_w = pad_to_multiple(image, divisor=16)
    input_image = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_logits = output.squeeze(0).cpu()
        pred = torch.sigmoid(pred_logits)[0].numpy()
        # Convert padded logits to binary mask and crop back
        mask_padded = (pred > 0.5).astype(np.uint8) * 255
        mask = mask_padded[:orig_h, :orig_w]
    return mask


def refine_mask(mask, kernel_size=(15,15)):
    """Refine the binary mask using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def isolate_road(image, mask):
    """Apply the mask to isolate road pixels."""
    return cv2.bitwise_and(image, image, mask=mask)


def crop_to_mask(image, mask, padding=20):
    """Crop image and mask around the non-zero mask region with padding."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    h, w = mask.shape[:2]
    minx = max(minx - padding, 0)
    miny = max(miny - padding, 0)
    maxx = min(maxx + padding, w - 1)
    maxy = min(maxy + padding, h - 1)
    return image[miny:maxy+1, minx:maxx+1], mask[miny:maxy+1, minx:maxx+1]


def overlay_mask(image, mask, color=(0, 255, 0), thickness=2):
    """Overlay mask edges on image."""
    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def preprocess_image(image):
    """Enhance contrast and reduce noise via CLAHE, gamma correction, and bilateral filtering."""
    # CLAHE on L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    img_cl = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Gamma correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype('uint8')
    img_gc = cv2.LUT(img_cl, table)
    # Bilateral filter
    img_bf = cv2.bilateralFilter(img_gc, 9, 75, 75)
    return img_bf


def main(data_dir, output_dir, model_type='unet', checkpoint=None, device='cuda'):
    """Process all images in the data directory to isolate roads."""
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_type, checkpoint, device)
    # Handle potential wrapper folder (e.g. data/coryell)
    root = Path(data_dir)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not (root / 'img').exists() and len(subdirs) == 1:
        print(f"Descending into wrapper folder '{subdirs[0].name}'")
        root = subdirs[0]
    # Process one image per road subfolder
    for road_dir in root.iterdir():
        if not road_dir.is_dir():
            continue
        img_folder = road_dir / 'img'
        if not img_folder.exists():
            continue
        img_files = sorted([p for p in img_folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        if not img_files:
            continue
        img_path = img_files[0]
        print(f"Processing road '{road_dir.name}' image: {img_path.name}")
        orig_image = cv2.imread(str(img_path))
        if orig_image is None:
            continue
        # Preprocess to enhance road features
        image = preprocess_image(orig_image)
        mask = segment_road(model, image, device)
        mask = refine_mask(mask)
        # Save full refined mask
        full_mask_path = os.path.join(output_dir, f"{road_dir.name.replace(' ', '_')}_full_mask.png")
        cv2.imwrite(full_mask_path, mask)
        print(f"Saved full mask: {full_mask_path}")
        # Full overlay on preprocessed image
        full_overlay = overlay_mask(image, mask)
        full_overlay_path = os.path.join(output_dir, f"{road_dir.name.replace(' ', '_')}_full_overlay.png")
        cv2.imwrite(full_overlay_path, full_overlay)
        print(f"Saved full overlay (preprocessed): {full_overlay_path}")
        # Crop around the road mask
        crop_res = crop_to_mask(image, mask, padding=20)
        if crop_res is None:
            print(f"No road detected for {road_dir.name}, skipping.")
            continue
        crop_img, crop_mask = crop_res
        out_name = road_dir.name.replace(' ', '_')
        crop_path = os.path.join(output_dir, f"{out_name}_crop.png")
        mask_path = os.path.join(output_dir, f"{out_name}_mask.png")
        cv2.imwrite(crop_path, crop_img)
        print(f"Saved crop image: {crop_path}")
        cv2.imwrite(mask_path, crop_mask)
        print(f"Saved crop mask: {mask_path}")
        # Save overlay of mask edges on the cropped image
        overlay = overlay_mask(crop_img, crop_mask)
        overlay_path = os.path.join(output_dir, f"{out_name}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved crop overlay: {overlay_path}")
    print("Processing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Isolate road from images")
    parser.add_argument('--data-dir', type=str, default='../data', help='Path to input images')
    parser.add_argument('--output-dir', type=str, default='preprocessing/output', help='Path to save outputs')
    parser.add_argument('--model', type=str, default='unet', choices=list(VALID_MODEL_TYPES), help='Segmentation model type')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.model, args.checkpoint, args.device) 
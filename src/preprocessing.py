import cv2
import os
from tqdm import tqdm

def resize_images(images, output_folder, size=(512, 512), filenames=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resized_images = []
    for idx, img in enumerate(tqdm(images, desc="Resizing images")):
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
        if filenames:
            out_path = os.path.join(output_folder, filenames[idx])
            cv2.imwrite(out_path, resized)
    return resized_images 
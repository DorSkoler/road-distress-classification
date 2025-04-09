import numpy as np
import matplotlib.pyplot as plt

def analyze_image_shapes(images):
    widths, heights = zip(*[(img.shape[1], img.shape[0]) for img in images])
    print(f"Total images: {len(images)}")
    print(f"Average Width: {np.mean(widths):.2f}px")
    print(f"Average Height: {np.mean(heights):.2f}px")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='skyblue')
    plt.title('Image Width Distribution')
    plt.xlabel('Width (px)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='salmon')
    plt.title('Image Height Distribution')
    plt.xlabel('Height (px)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show() 
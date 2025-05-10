import os
import shutil

# Source and destination directories
src_dir = 'preprocessing/annotations'
dst_dir = 'preprocessing/output'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Move all *_crop.png files
for filename in os.listdir(src_dir):
    if filename.endswith('_crop.png'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved {filename} to {dst_dir}") 
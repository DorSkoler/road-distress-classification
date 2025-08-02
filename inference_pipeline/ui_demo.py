#!/usr/bin/env python3
"""
Demo Script for UI Testing
Date: 2025-08-01

Creates sample images for testing the UI when no real road images are available.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path

def create_sample_road_images():
    """Create sample road images for UI testing."""
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    # Image configurations
    configs = [
        {"name": "good_road", "damage": False, "description": "Good road condition"},
        {"name": "cracked_road", "damage": True, "description": "Road with cracks"},
        {"name": "pothole_road", "damage": True, "description": "Road with potholes"},
        {"name": "partially_damaged", "damage": True, "description": "Partially damaged road"},
        {"name": "clear_road", "damage": False, "description": "Clear road surface"},
    ]
    
    print("Creating sample road images for UI testing...")
    
    for config in configs:
        # Create base road image
        img = create_road_image(config["damage"], config["description"])
        
        # Save image
        img_path = output_dir / f"{config['name']}.jpg"
        img.save(img_path, quality=95)
        print(f"Created: {img_path}")
    
    print(f"\nSample images created in: {output_dir}")
    print("You can now test the UI with these images!")
    
    return output_dir

def create_road_image(has_damage=False, description="Road"):
    """Create a synthetic road image."""
    width, height = 800, 600
    
    # Create base image with road-like colors
    img = Image.new('RGB', (width, height), color=(70, 70, 70))  # Dark gray road
    draw = ImageDraw.Draw(img)
    
    # Add road markings
    # Center line
    for y in range(0, height, 40):
        draw.rectangle([width//2 - 5, y, width//2 + 5, y + 20], fill=(255, 255, 255))
    
    # Side lines
    draw.rectangle([50, 0, 60, height], fill=(255, 255, 255))
    draw.rectangle([width-60, 0, width-50, height], fill=(255, 255, 255))
    
    # Add some texture/noise to make it look more realistic
    for _ in range(1000):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        color = random.randint(60, 80)
        draw.point([x, y], fill=(color, color, color))
    
    if has_damage:
        # Add damage patterns
        add_damage_patterns(draw, width, height)
    
    # Add some roadside elements
    add_roadside_elements(draw, width, height)
    
    # Add text label
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), description, fill=(255, 255, 0), font=font)
    except:
        draw.text((10, 10), description, fill=(255, 255, 0))
    
    return img

def add_damage_patterns(draw, width, height):
    """Add damage patterns to the road image."""
    # Add cracks
    for _ in range(random.randint(3, 8)):
        start_x = random.randint(100, width-100)
        start_y = random.randint(100, height-100)
        
        # Create crack pattern
        points = [(start_x, start_y)]
        for i in range(random.randint(20, 50)):
            last_x, last_y = points[-1]
            new_x = last_x + random.randint(-10, 10)
            new_y = last_y + random.randint(-5, 15)
            
            if 50 < new_x < width-50 and 0 < new_y < height:
                points.append((new_x, new_y))
        
        # Draw crack
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=(40, 40, 40), width=random.randint(2, 5))
    
    # Add potholes
    for _ in range(random.randint(1, 3)):
        x = random.randint(100, width-150)
        y = random.randint(100, height-150)
        w = random.randint(30, 80)
        h = random.randint(20, 60)
        
        # Draw pothole (darker ellipse)
        draw.ellipse([x, y, x+w, y+h], fill=(30, 30, 30))
        # Add shadow effect
        draw.ellipse([x+2, y+2, x+w+2, y+h+2], outline=(20, 20, 20), width=2)

def add_roadside_elements(draw, width, height):
    """Add roadside elements for realism."""
    # Add some grass/dirt on sides
    for side in [0, width-30]:
        for y in range(0, height, 20):
            color = random.randint(40, 60)
            draw.rectangle([side, y, side+30, y+15], 
                         fill=(color//2, color, color//2))  # Greenish
    
    # Add some random debris
    for _ in range(random.randint(5, 15)):
        x = random.randint(60, width-60)
        y = random.randint(0, height)
        size = random.randint(2, 8)
        color = random.randint(80, 120)
        draw.ellipse([x, y, x+size, y+size], fill=(color, color, color))

def main():
    """Main function to create sample images."""
    print("🛣️ Road Distress UI Demo Setup")
    print("=" * 40)
    
    # Create sample images
    sample_dir = create_sample_road_images()
    
    print("\n📋 How to test the UI:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Launch UI: python launch_ui.py")
    print("3. Upload the sample images from:", sample_dir)
    print("4. Test both single image and batch processing modes")
    
    print("\n🎯 What to expect:")
    print("- good_road.jpg & clear_road.jpg should show NO DAMAGE")
    print("- cracked_road.jpg, pothole_road.jpg & partially_damaged.jpg should show DAMAGE")
    print("- Each image will generate confidence heatmaps")
    print("- Batch processing will show summary statistics")

if __name__ == "__main__":
    main()
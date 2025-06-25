#!/usr/bin/env python3
"""
Create synthetic photoshopped images for testing
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

def create_base_image(size=(300, 300), color=(100, 150, 200)):
    """Create a base image with some content"""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Add some simple shapes/text to make it more interesting
    draw.rectangle([50, 50, 250, 250], fill=(200, 100, 150), outline=(255, 255, 255), width=3)
    draw.ellipse([100, 100, 200, 200], fill=(100, 200, 100))
    draw.text((120, 140), "TEST", fill=(255, 255, 255))
    
    return img

def create_photoshopped_images():
    """Create synthetic photoshopped images"""
    print("🎨 Creating synthetic photoshopped images...")
    
    # Create directories
    base_dir = "photoshopped_images"
    original_dir = os.path.join(base_dir, "original")
    edited_dir = os.path.join(base_dir, "edited")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)
    
    # Create 5 base images
    for i in range(5):
        print(f"\n📸 Creating image set {i+1}/5...")
        
        # Create base image
        base_img = create_base_image()
        original_path = os.path.join(original_dir, f"original_{i}.png")
        base_img.save(original_path)
        print(f"✅ Saved original: original_{i}.png")
        
        # Create various "photoshopped" versions
        edits = [
            ("brightness", lambda x: ImageEnhance.Brightness(x).enhance(1.8), "Increased brightness"),
            ("contrast", lambda x: ImageEnhance.Contrast(x).enhance(1.5), "Enhanced contrast"),
            ("blur", lambda x: x.filter(ImageFilter.BLUR), "Applied blur effect"),
            ("sharpen", lambda x: x.filter(ImageFilter.SHARPEN), "Sharpened image"),
            ("color_boost", lambda x: ImageEnhance.Color(x).enhance(2.0), "Boosted colors"),
            ("saturation", lambda x: ImageEnhance.Color(x).enhance(0.5), "Reduced saturation"),
            ("noise", lambda x: add_noise(x), "Added noise"),
            ("vintage", lambda x: apply_vintage_effect(x), "Vintage filter"),
        ]
        
        for edit_name, edit_func, description in edits:
            try:
                edited_img = edit_func(base_img.copy())
                edited_filename = f"photoshopped_{edit_name}_{i}.png"
                edited_path = os.path.join(edited_dir, edited_filename)
                edited_img.save(edited_path)
                print(f"✅ Created: {edited_filename} ({description})")
            except Exception as e:
                print(f"❌ Failed to create {edit_name}: {e}")
    
    print(f"\n🎉 Successfully created photoshopped images!")
    print(f"📁 Original images: {original_dir}")
    print(f"📁 Edited images: {edited_dir}")

def add_noise(img):
    """Add noise to image"""
    img_array = np.array(img)
    noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_vintage_effect(img):
    """Apply vintage/sepia effect"""
    # Convert to sepia
    img_array = np.array(img)
    sepia = np.array([0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131]).reshape(3, 3)
    sepia_img = np.dot(img_array, sepia.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    # Add some vintage coloring
    vintage_img = Image.fromarray(sepia_img)
    vintage_img = ImageEnhance.Color(vintage_img).enhance(0.8)
    vintage_img = ImageEnhance.Brightness(vintage_img).enhance(0.9)
    
    return vintage_img

if __name__ == "__main__":
    create_photoshopped_images() 
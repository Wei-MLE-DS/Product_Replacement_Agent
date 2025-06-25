#!/usr/bin/env python3
"""
Create a large dataset for image authenticity detection
Focus on photoshopped and AI-generated images
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from scipy.ndimage import gaussian_filter
import random

def create_base_images(count=100):
    """Create diverse base images for photoshopping"""
    print(f"🎨 Creating {count} base images...")
    
    base_dir = "large_dataset/base_images"
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(count):
        # Create different types of base images
        img_type = i % 5
        
        if img_type == 0:
            # Simple colored rectangles
            img = Image.new('RGB', (256, 256), color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 200, 200], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            
        elif img_type == 1:
            # Gradient images
            img_array = np.random.rand(256, 256, 3)
            img_array = gaussian_filter(img_array, sigma=2)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            
        elif img_type == 2:
            # Geometric patterns
            img = Image.new('RGB', (256, 256), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)
            for j in range(10):
                x1, y1 = random.randint(0, 256), random.randint(0, 256)
                x2, y2 = random.randint(0, 256), random.randint(0, 256)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.line([x1, y1, x2, y2], fill=color, width=3)
                
        elif img_type == 3:
            # Text-based images
            img = Image.new('RGB', (256, 256), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)
            draw.text((50, 100), f"TEXT {i}", fill=(255, 255, 255))
            draw.text((50, 150), "SAMPLE", fill=(255, 255, 0))
            
        else:
            # Noise-based images
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
        
        filename = f"base_{i:03d}.png"
        save_path = os.path.join(base_dir, filename)
        img.save(save_path)
        
        if i % 20 == 0:
            print(f"  ✅ Created {i+1}/{count} base images")
    
    print(f"📊 Created {count} base images")
    return base_dir

def create_photoshopped_images(base_dir, count=500):
    """Create a large number of photoshopped images"""
    print(f"\n🎨 Creating {count} photoshopped images...")
    
    photoshopped_dir = "large_dataset/photoshopped"
    os.makedirs(photoshopped_dir, exist_ok=True)
    
    # Get list of base images
    base_files = [f for f in os.listdir(base_dir) if f.endswith('.png')]
    
    # Define various photoshop effects
    effects = [
        ("brightness", lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.3, 2.5))),
        ("contrast", lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.3, 2.5))),
        ("color", lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.3, 2.5))),
        ("saturation", lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.1, 3.0))),
        ("blur", lambda x: x.filter(ImageFilter.BLUR)),
        ("sharpen", lambda x: x.filter(ImageFilter.SHARPEN)),
        ("edge_enhance", lambda x: x.filter(ImageFilter.EDGE_ENHANCE)),
        ("emboss", lambda x: x.filter(ImageFilter.EMBOSS)),
        ("find_edges", lambda x: x.filter(ImageFilter.FIND_EDGES)),
        ("smooth", lambda x: x.filter(ImageFilter.SMOOTH)),
        ("noise", lambda x: add_noise(x)),
        ("vintage", lambda x: apply_vintage_effect(x)),
        ("sepia", lambda x: apply_sepia_effect(x)),
        ("posterize", lambda x: apply_posterize_effect(x)),
        ("solarize", lambda x: apply_solarize_effect(x)),
    ]
    
    created_count = 0
    
    for i in range(count):
        # Select random base image
        base_file = random.choice(base_files)
        base_path = os.path.join(base_dir, base_file)
        base_img = Image.open(base_path).copy()
        
        # Apply 1-3 random effects
        num_effects = random.randint(1, 3)
        selected_effects = random.sample(effects, num_effects)
        
        edited_img = base_img
        effect_names = []
        
        for effect_name, effect_func in selected_effects:
            try:
                edited_img = effect_func(edited_img)
                effect_names.append(effect_name)
            except Exception as e:
                continue
        
        # Save the edited image
        filename = f"photoshopped_{i:04d}_{'_'.join(effect_names)}.png"
        save_path = os.path.join(photoshopped_dir, filename)
        edited_img.save(save_path)
        
        created_count += 1
        
        if i % 50 == 0:
            print(f"  ✅ Created {i+1}/{count} photoshopped images")
    
    print(f"📊 Created {created_count} photoshopped images")
    return created_count

def create_ai_generated_images(count=500):
    """Create a large number of AI-generated images"""
    print(f"\n🤖 Creating {count} AI-generated images...")
    
    ai_dir = "large_dataset/ai_generated"
    os.makedirs(ai_dir, exist_ok=True)
    
    created_count = 0
    
    for i in range(count):
        # Different AI generation methods
        method = i % 8
        
        if method == 0:
            # Smooth gradients (common in AI art)
            img = create_smooth_gradient()
        elif method == 1:
            # Fractal-like patterns
            img = create_fractal_pattern()
        elif method == 2:
            # Artistic noise
            img = create_artistic_noise()
        elif method == 3:
            # Abstract shapes
            img = create_abstract_shapes()
        elif method == 4:
            # Neural-style patterns
            img = create_neural_style_pattern()
        elif method == 5:
            # Dream-like images
            img = create_dream_like_image()
        elif method == 6:
            # Geometric AI art
            img = create_geometric_ai_art()
        else:
            # Mixed AI styles
            img = create_mixed_ai_style()
        
        # Apply AI-like post-processing
        img = apply_ai_post_processing(img)
        
        # Save the image
        filename = f"ai_generated_{i:04d}.png"
        save_path = os.path.join(ai_dir, filename)
        img.save(save_path)
        
        created_count += 1
        
        if i % 50 == 0:
            print(f"  ✅ Created {i+1}/{count} AI-generated images")
    
    print(f"📊 Created {created_count} AI-generated images")
    return created_count

def create_smooth_gradient():
    """Create smooth gradient patterns"""
    size = (256, 256)
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    X, Y = np.meshgrid(x, y)
    
    # Complex gradients
    r = np.sin(X * random.uniform(1, 5) + Y * random.uniform(1, 5)) * 0.5 + 0.5
    g = np.cos(X * random.uniform(1, 5) + Y * random.uniform(1, 5)) * 0.5 + 0.5
    b = np.sin(X * random.uniform(1, 5) + Y * random.uniform(1, 5)) * 0.5 + 0.5
    
    img_array = np.stack([r, g, b], axis=2) * 255
    return Image.fromarray(img_array.astype(np.uint8))

def create_fractal_pattern():
    """Create fractal-like patterns"""
    size = (256, 256)
    x = np.linspace(-2, 2, size[0])
    y = np.linspace(-2, 2, size[1])
    X, Y = np.meshgrid(x, y)
    
    Z = X + Y*1j
    C = Z.copy()
    fractal = np.zeros(Z.shape)
    
    for _ in range(15):
        Z = Z**2 + C
        fractal += (np.abs(Z) < 2).astype(float)
    
    fractal = fractal / fractal.max()
    img_array = np.stack([fractal, fractal*0.7, fractal*0.3], axis=2) * 255
    return Image.fromarray(img_array.astype(np.uint8))

def create_artistic_noise():
    """Create artistic noise patterns"""
    size = (256, 256)
    noise = np.random.rand(size[0], size[1], 3)
    
    # Apply multiple layers of smoothing
    for _ in range(3):
        noise = gaussian_filter(noise, sigma=1)
    
    # Add color variations
    noise[:, :, 0] *= random.uniform(0.5, 1.5)
    noise[:, :, 1] *= random.uniform(0.5, 1.5)
    noise[:, :, 2] *= random.uniform(0.5, 1.5)
    
    img_array = (noise * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def create_abstract_shapes():
    """Create abstract shape patterns"""
    img = Image.new('RGB', (256, 256), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    for _ in range(random.randint(10, 30)):
        shape_type = random.choice(['circle', 'rectangle', 'line'])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        if shape_type == 'circle':
            x, y = random.randint(0, 256), random.randint(0, 256)
            radius = random.randint(10, 50)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        elif shape_type == 'rectangle':
            x1, y1 = random.randint(0, 256), random.randint(0, 256)
            x2, y2 = random.randint(0, 256), random.randint(0, 256)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            x1, y1 = random.randint(0, 256), random.randint(0, 256)
            x2, y2 = random.randint(0, 256), random.randint(0, 256)
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
    
    return img

def create_neural_style_pattern():
    """Create neural-style transfer-like patterns"""
    img_array = np.random.rand(256, 256, 3)
    
    # Apply different filters to each channel
    img_array[:, :, 0] = gaussian_filter(img_array[:, :, 0], sigma=2)
    img_array[:, :, 1] = gaussian_filter(img_array[:, :, 1], sigma=1)
    img_array[:, :, 2] = gaussian_filter(img_array[:, :, 2], sigma=3)
    
    # Add some texture
    texture = np.random.rand(256, 256) * 0.3
    for i in range(3):
        img_array[:, :, i] += texture
    
    img_array = np.clip(img_array, 0, 1) * 255
    return Image.fromarray(img_array.astype(np.uint8))

def create_dream_like_image():
    """Create dream-like, surreal images"""
    img_array = np.random.rand(256, 256, 3)
    
    # Create flowing patterns
    for _ in range(5):
        img_array = gaussian_filter(img_array, sigma=1.5)
        img_array += np.random.rand(256, 256, 3) * 0.1
    
    # Add dreamy colors
    img_array[:, :, 0] *= 1.2  # More red
    img_array[:, :, 1] *= 0.8  # Less green
    img_array[:, :, 2] *= 1.1  # More blue
    
    img_array = np.clip(img_array, 0, 1) * 255
    return Image.fromarray(img_array.astype(np.uint8))

def create_geometric_ai_art():
    """Create geometric AI art patterns"""
    img = Image.new('RGB', (256, 256), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    # Create geometric patterns
    for i in range(20):
        points = []
        for _ in range(random.randint(3, 6)):
            points.append((random.randint(0, 256), random.randint(0, 256)))
        
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.polygon(points, fill=color, outline=(255, 255, 255), width=1)
    
    return img

def create_mixed_ai_style():
    """Create mixed AI art styles"""
    # Combine multiple techniques
    img1 = create_smooth_gradient()
    img2 = create_artistic_noise()
    
    # Blend them together
    img1_array = np.array(img1).astype(float)
    img2_array = np.array(img2).astype(float)
    
    blend = img1_array * 0.6 + img2_array * 0.4
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blend)

def apply_ai_post_processing(img):
    """Apply AI-like post-processing"""
    # Enhance colors slightly
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    
    return img

def add_noise(img):
    """Add noise to image"""
    img_array = np.array(img)
    noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_vintage_effect(img):
    """Apply vintage effect"""
    img_array = np.array(img)
    sepia = np.array([0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131]).reshape(3, 3)
    sepia_img = np.dot(img_array, sepia.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    vintage_img = Image.fromarray(sepia_img)
    vintage_img = ImageEnhance.Color(vintage_img).enhance(0.8)
    vintage_img = ImageEnhance.Brightness(vintage_img).enhance(0.9)
    
    return vintage_img

def apply_sepia_effect(img):
    """Apply sepia effect"""
    img_array = np.array(img)
    sepia = np.array([0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131]).reshape(3, 3)
    sepia_img = np.dot(img_array, sepia.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img)

def apply_posterize_effect(img):
    """Apply posterize effect"""
    return img.quantize(colors=8).convert('RGB')

def apply_solarize_effect(img):
    """Apply solarize effect"""
    img_array = np.array(img)
    solarized = np.where(img_array < 128, img_array, 255 - img_array)
    return Image.fromarray(solarized)

def main():
    print("🎯 Creating Large Image Authenticity Detection Dataset")
    print("Focus: Photoshopped and AI-Generated Images")
    
    # Create base images
    base_dir = create_base_images(count=100)
    
    # Create photoshopped images
    photoshopped_count = create_photoshopped_images(base_dir, count=500)
    
    # Create AI-generated images
    ai_count = create_ai_generated_images(count=500)
    
    # Create summary
    print(f"\n🎉 Large dataset creation completed!")
    print(f"📊 Summary:")
    print(f"  Base images: 100")
    print(f"  Photoshopped images: {photoshopped_count}")
    print(f"  AI-generated images: {ai_count}")
    print(f"  Total: {100 + photoshopped_count + ai_count}")
    print(f"📁 Dataset location: large_dataset/")
    
    # Create dataset info file
    with open("large_dataset/dataset_info.txt", "w") as f:
        f.write("Large Image Authenticity Detection Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Base images: 100\n")
        f.write(f"Photoshopped images: {photoshopped_count}\n")
        f.write(f"AI-generated images: {ai_count}\n")
        f.write(f"Total images: {100 + photoshopped_count + ai_count}\n\n")
        f.write("Directory structure:\n")
        f.write("large_dataset/\n")
        f.write("  ├── base_images/     # Original base images\n")
        f.write("  ├── photoshopped/    # Edited/photoshopped images\n")
        f.write("  ├── ai_generated/    # AI-generated images\n")
        f.write("  └── dataset_info.txt # This file\n")

if __name__ == "__main__":
    main() 
import os
from PIL import Image
import numpy as np

base_path = r'dataset'

# Create sample images for each split and class
for split in ['train', 'val', 'test']:
    for class_name in ['class1', 'class2']:
        class_path = os.path.join(base_path, split, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        # Create 5 random images per class
        num_images = 5
        for i in range(num_images):
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(class_path, f'image_{i:03d}.jpg')
            img.save(img_path)
            print(f'Created: {img_path}')

print('\n✓ Sample dataset created successfully!')

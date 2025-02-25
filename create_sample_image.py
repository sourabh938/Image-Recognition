"""
Create a sample image for testing the image classifier
"""

from PIL import Image, ImageDraw
import numpy as np

# Create a 300x300 RGB image with a gradient background
width, height = 300, 300
image = Image.new('RGB', (width, height), color=(255, 255, 255))
draw = ImageDraw.Draw(image)

# Create a gradient background
for y in range(height):
    for x in range(width):
        r = int(255 * (1 - x / width))
        g = int(255 * (y / height))
        b = int(255 * (0.5 + 0.5 * np.sin(x/30) * np.cos(y/30)))
        draw.point((x, y), fill=(r, g, b))

# Draw some shapes
# Red circle
draw.ellipse((50, 50, 150, 150), fill=(255, 0, 0))
# Blue rectangle
draw.rectangle((180, 50, 250, 120), fill=(0, 0, 255))
# Green triangle
draw.polygon([(100, 180), (180, 250), (220, 180)], fill=(0, 255, 0))

# Save the image
image.save('sample_image.jpg')
print("Sample image created: sample_image.jpg") 
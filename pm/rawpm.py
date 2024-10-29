import numpy as np
from PIL import Image
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

# Load the image
image_path = 'test.png'
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_np = np.array(image)

# Compute the image gradient using Sobel filter
Ix = sobel(image_np, axis=0)  # Gradient in x
Iy = sobel(image_np, axis=1)  # Gradient in y

# Compute the photometric moments
m00 = np.sum(image_np)  # Zeroth moment
m10 = np.sum(Ix * image_np)  # First moment in x
m01 = np.sum(Iy * image_np)  # First moment in y
m11 = np.sum(Ix * Iy)  # Cross moment
m20 = np.sum(Ix**2 * image_np)  # Second moment in x
m02 = np.sum(Iy**2 * image_np)  # Second moment in y

# Store the moments in a dictionary
photometric_moments = {'m00': m00, 'm10': m10, 'm01': m01, 'm11': m11, 'm20': m20, 'm02': m02}

# Visualize the photometric moments
moments_names = list(photometric_moments.keys())
moments_values = list(photometric_moments.values())

plt.figure(figsize=(10, 5))
plt.bar(moments_names, moments_values, color='blue')
plt.title('Photometric Moments')
plt.xlabel('Moments')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

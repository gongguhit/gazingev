import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Photometric Gaussian function
def photometric_gaussian(u_g, u, v, I_u, lambda_g):
    return np.exp(-((u_g[0] - u)**2 + (u_g[1] - v)**2) / (2 * (lambda_g * I_u)**2))

# Gaussian mixture function
def gaussian_mixture(u_g, image, lambda_g):
    height, width = image.shape
    mixture_sum = 0
    
    for u in range(height):
        for v in range(width):
            if image[u, v] > 0:  # Ignore black pixels
                I_u = image[u, v]
                g_value = photometric_gaussian(u_g, u, v, I_u, lambda_g)
                mixture_sum += g_value
    
    return mixture_sum

# Function to simulate Gaussian mixture for the entire image
def simulate_gaussian_mixture(image, lambda_g):
    height, width = image.shape
    G_mixture = np.zeros((height, width))
    
    for u in range(height):
        for v in range(width):
            u_g = (u, v)
            g_value = gaussian_mixture(u_g, image, lambda_g)
            G_mixture[u, v] = g_value
    
    return G_mixture

# Load the image
image_path = './test.png'
image = Image.open(image_path).convert('L')
image = np.array(image)

# Define lambda_g
lambda_g = 0.1  # Adjust this value as needed

# Visualize the result
result = simulate_gaussian_mixture(image, lambda_g)

plt.imshow(result, cmap='jet')
plt.colorbar()
plt.title('Gaussian Mixture Simulation')
plt.show()

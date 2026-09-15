#######################################
# # # Imports # # #
# PIL : Image Manip.
# numpy : Image Accessing
# mat : plots
#######################################

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

#######################################
# # # load_grayscale # # #
# Use convert 'L' to convert to grayscale
# Atkinson alg. is a single channel alg so each pixel must be in range of 0-1 (black to white)

# img should be in float since we will be conducting some maths
#######################################

def load_grayscale(path, target_width=None):
    img = Image.open(path).convert('L')

    if target_width is not None:
        img = resize(img, target_width)

    return np.array(img, dtype=np.float64)

#######################################
# # # Resize # # #
#Image.LANCZOS applies resampling filter (compute new pixel values)
#######################################

def resize(img, target_width):

    w, h = img.size

    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)

    return img.resize((target_width, target_height), Image.LANCZOS)

#######################################
# # # Atkinson Dither # # #

# param :
    # img_aray : np image arr
    # threshold :  uh threshold
#######################################

def atkinson_dither(img_array, threshold=128):

    img = img_array.copy()
    h, w = img.shape

    #Atkinson diffusion kernel (* = pixel being scanned)
    # _   _   *   1/8   1/8
    # .  1/8 1/8  1/8   .
    # .  .   1/8   .    .

    diffusion_pattern = [
        (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2)
    ]

    for y in range(h):
        for x in range(w):

            og_val = img[y, x]

            # Thresholding:

            if og_val >= threshold:
                new_val = 255
            else:
                new_val = 0

            img[y, x] = new_val

            #Error (difference of change DELTA):
            # 6 neighbors gets the change divided by 8, 6/8 = 75% error gets trans. only

            error = og_val - new_val
            diffused_error = error / 8


            # Push error onto kernel:

            for dx, dy in diffusion_pattern:

                #relative pixel to real coord
                nx, ny = x + dx, y + dy

                #add error to real coord pixels
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] += diffused_error

    #clip = rounding to correct range
    #also needa convert back to int
    return np.clip(img, 0, 255).astype(np.uint8)

#######################################
# # # Comparison # # #
#######################################

def show_comp(original, dithered, title="Atkinson Dithering"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(dithered, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

#######################################
# # # MAIN # # #
#######################################

path = "mountain.png"
og = load_grayscale(path, 255)

dithered = atkinson_dither(og)
show_comp(og, dithered)
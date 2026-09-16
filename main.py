#######################################
# # # Imports # # #
# PIL : Image Manip.
# numpy : Image Accessing
# mat : plots
#######################################

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import os

#VIDEO

import imageio
import re


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

    #Atkinson diffusion kernel
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

def show_comp(og, mod, title="Modified"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(og, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(mod, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

#######################################
# # # save # # #
# saves a numpy image arry to path
#######################################

def save_image(img_array, path):
    Image.fromarray(img_array).save(path)

#######################################
# # # Atkinson Dither # # #
#######################################

def atkinson_dither_output(img_array, threshold=128):

    # img = img_array.copy()
    img = img_array.astype(np.float64)
    h, w = img.shape

    diffusion_pattern = [
        (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2)
    ]

    for y in range(h):
        for x in range(w):

            og_val = img[y, x]

            if og_val >= threshold:
                new_val = 255
            else:
                new_val = 0

            img[y, x] = new_val

            error = og_val - new_val
            diffused_error = error / 8

            for dx, dy in diffusion_pattern:

                nx, ny = x + dx, y + dy

                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] += diffused_error

        if y % 10 == 0:
            save_dir = "output/images"
            os.makedirs(save_dir, exist_ok=True)

            filename = f"output{y}.png"

            pathOutput = os.path.join(save_dir, filename)
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(pathOutput)

    return np.clip(img, 0, 255).astype(np.uint8)


#######################################
# # # MAIN # # #
#######################################

# path = "mountain.png"
# pathDith = "mountainDither.png"

path = "gato.png"
pathDith = "gatoDith.png"

og = load_grayscale(path, 250)
dithered = atkinson_dither(og)

show_comp(og, dithered)
save_image(dithered, pathDith)

################################
# SAVE ANIMATION:

# dithered_Final = atkinson_dither_output(og)

#IMAGE-VIDEO THING

# save_dir = "output/images"
#
# files = os.listdir(save_dir)
# files = [f for f in files if f.startswith("output") and f.endswith(".png")]
# files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
#
# writer = imageio.get_writer("output/dither_progress.mp4", fps=30)
#
# for f in files:
#     frame = imageio.imread(os.path.join(save_dir, f))
#     writer.append_data(frame)
#     writer.append_data(frame)
#     writer.append_data(frame)
#     writer.append_data(frame)
#
# writer.close()
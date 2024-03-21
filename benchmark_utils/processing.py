import numpy as np


def grey_image(colored_img):
    n, m, _ = np.shape(colored_img)
    new_img = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            new_img[i][j] = 1/3 * (colored_img[0] +
                                   colored_img[1] +
                                   colored_img[2])

    return new_img


def flatten_images(images):
    # Initialize an empty list to store flattened images
    X_flatten = []

    # Iterate over each image
    for img in images:
        # Flatten the image and append it to the list
        flattened_img = img.flatten()
        X_flatten.append(flattened_img)

    # Convert the list of flattened images to a NumPy array
    return np.array(X_flatten)

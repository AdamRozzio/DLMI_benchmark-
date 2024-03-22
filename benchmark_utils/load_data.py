import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_images_from_folder(folder_path):
    images = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a regular file and has a JPEG extension
        if os.path.isfile(file_path) and file_path.lower().endswith('.jpg'):
            # Open the image using PIL's Image.open
            with Image.open(file_path) as image:
                # Convert the image to numpy array and append to the list
                images.append(np.transpose(np.array(image)))

    return images


def load_data(csv_path, images_path):
    # Load the CSV file using pandas
    dataframe = pd.read_csv(csv_path)

    # Create a dictionary to store the information
    data_dict = {}

    # Iterate through the rows of the DataFrame .
    # and add the informationto the dictionary
    for index, row in dataframe.iterrows():
        # Make sure to adjust the column names
        # based on your actual CSV structure
        label = row['LABEL']
        date_of_birth = row['DOB']
        gender = row['GENDER']
        identifier = row['ID']
        lymphocyte_count = row['LYMPH_COUNT']
        img_path = images_path + identifier
        images = load_images_from_folder(img_path)

        # Add the information to the dictionary
        data_dict[index] = {
            'label': label,
            'dob': date_of_birth,
            'gender': gender,
            'id': identifier,
            'lymph_count': lymphocyte_count,
            'images': images
        }

    return data_dict


def rgb_to_grayscale(rgb_image):
    # Convert RGB to grayscale using the luminosity method
    grayscale_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    # Convert to uint8 data type (required for Pillow)
    grayscale_image = grayscale_image.astype(np.uint8)

    return grayscale_image


def load_X_y(data):
    X = []
    y = []
    for j in range(15):
        images_subject_j = data[j]['images']
        for i in range(len(images_subject_j)):
            X.append(data[j]['images'][i])
            y.append(data[j]['label'])
        print("loading of image:", j)

    X = np.array(X)
    y = np.array(y)

    return X, y


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'image': self.X[idx], 'label': self.y[idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

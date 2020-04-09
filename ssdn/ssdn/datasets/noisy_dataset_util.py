import os
from PIL import Image
import cv2 as cv
import torch

transform_to_tensor = torchvision.transforms.ToTensor()
transform_to_image = torchvision.transforms.ToPILImage()

def noisy_train(original_path: str, target_path: str):
    """Given the path to a folder containing a set of images,
    create a set of noisy images under the target folder.
    Uses Gaussian noise for now but can change but cba.

    Args:
        original_path (str): Path of folder of original images.
        target_path (str): Path of folder where noisy images will be saved.
    """
    count = 0
    for r, d, f in os.walk(original_path):
        for file in f:
            file_path = os.path.join(r, file)
            np_img = cv.imread(file_path) / 255
            img = torch.from_numpy(np_img)
            noisy_img = add_gaussian_noise(img, 25).numpy() * 255
            # Now save image to google drive as noisy training data
            cv.imwrite(target_path + file, noisy_img)
            count += 1
            print(f'{count} of {len(f)} images complete')

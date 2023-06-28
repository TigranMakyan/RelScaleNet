import torch
import numpy as np
import cv2

class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))

    def create_mask(self, img_arr: np.ndarray, mask_size: int):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        return edges

    def process(self, img_arr: np.ndarray, mask_size: int):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        img_arr = cv2.resize(img_arr, (mask_size, mask_size))
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(edges).to(torch.bool).unsqueeze(dim=0)  # (1,H,W)
        # print('In process ing: ', img_tensor_norm.shape)
        # print('In process mask: ', amask_tensor.shape)
        return img_tensor_norm, amask_tensor

# processor = PreprocessorX()
# image = cv2.imread('/home/user/Desktop/image.jpg')
# a, b = processor.process(image, 320)
# print(a.shape, b.shape)
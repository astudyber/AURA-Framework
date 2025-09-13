import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

class Denoise(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(Denoise, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=1, bias=False))
        self.Denoise = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.Denoise(x)
        return x - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Load pre trained model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Denoise().to(device)
    model = torch.load(os.path.join(model_path))
    model.eval()
    return model

# Test image denoising
def test_image(model, image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')  # 转换为RGB图像
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Separate RGB channels
    r, g, b = image_tensor[:, 0:1, :, :], image_tensor[:, 1:2, :, :], image_tensor[:, 2:3, :, :]

    # Denoising each channel separately
    with torch.no_grad():
        denoised_r = model(r)
        denoised_g = model(g)
        denoised_b = model(b)

    # Merge denoised channels
    denoised_image = torch.cat((denoised_r, denoised_g, denoised_b), dim=1)
    denoised_image = denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    denoised_image = np.clip(denoised_image, 0, 1) * 255
    denoised_image = denoised_image.astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))
    print(f"The denoised image has been saved to {output_path}")

# Process all images in the folder
def process_folder(input_folder, output_folder, model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = load_model(model_path)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            test_image(model, input_path, output_path)

if __name__ == "__main__":
    model_path = "model.pth"  # Replace with your pre trained model path
    input_folder = "./"  # Replace with your folder path
    output_folder = "./1/"  # Replace with your folder path

    process_folder(input_folder, output_folder, model_path)
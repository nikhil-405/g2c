import numpy as np
import requests
from io import BytesIO
from PIL import Image
import torch 
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def reconstruct(L_sample, probs):
    cluster_centers = np.load(r"assets\ab_clusters.npy")

    L_np = L_sample.detach().cpu().numpy() # [B, 1, H, W]
    probs_np = probs.detach().cpu().numpy() # [B, 313, H, W]

    batch_size, _, H, W = L_np.shape
    ab_images = np.zeros((batch_size, H, W, 2), dtype = np.float32)

    for i in range(batch_size):
        prob_i = probs_np[i].reshape(313, -1) # [313, H*W]

        # expected value over ab centers: [2, H*W]
        ab_flat = np.dot(cluster_centers.T, prob_i)  # [2, H*W]
        ab = ab_flat.T.reshape(H, W, 2) # [H, W, 2]
        ab_images[i] = ab

    # (L, ab) --> RGB
    rgb_images = []
    for i in range(batch_size):
        L = L_np[i, 0] * 50.0 + 50.0
        Lab = np.zeros((H, W, 3), dtype=np.float32)
        Lab[:, :, 0] = L
        Lab[:, :, 1:] = ab_images[i] * 128.0 + 128.0  # de-normalize ab

        # to RGB
        Lab = Lab.astype(np.uint8)
        rgb = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb_images.append(rgb)

    rgb_images = np.stack(rgb_images, axis = 0) # [B, H, W, 3]
    return rgb_images


def infer_and_show(image_url, model, reconstruct_fn, device, img_size = (256, 256)):
    # 1) download image
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error in loading image")
        return

    # 2) resize and to numpy
    img_resized = img.resize(img_size)
    img_np = np.array(img_resized)
    
    # 3) LAB and normalize
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L_norm = (L / 50.0) - 1.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    # 4) infer
    model.eval()
    with torch.no_grad():
        logits = model(L_tensor) # [1, 313, H, W]
        probs = F.softmax(logits, dim=1)
        rgb_image = reconstruct_fn(L_tensor, probs)  # [1, H, W, 3]

    # 5) display
    L_gray = (L_tensor.repeat(1, 3, 1, 1).squeeze(0).permute(1, 2, 0) + 1.0) / 2.0  # [H, W, 3]
    rgb_image = rgb_image[0]  # [H, W, 3]

    plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(L_gray.cpu().numpy())
    plt.title("Input Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.title("Colorized Output")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
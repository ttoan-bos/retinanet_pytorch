import torch
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_orig = image.copy()

    rows, cols, _ = image.shape
    smallest_side = min(rows, cols)

    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    image = cv2.resize(
        image,
        (int(round(cols * scale)), int(round(rows * scale)))
    )

    rows, cols, _ = image.shape
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, 3), dtype=np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    image = new_image / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]

    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)   # B=1

    return torch.from_numpy(image).float(), scale, image_orig

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("model_final_224.pt", map_location=device, weights_only=False)

if hasattr(model, "module"):
    model = model.module

model = model.to(device)
model.eval()

torch.save(model.state_dict(), "model_weight.pth")
print("Save ok!")




img, scale, img_orig = preprocess_image("C:\\Users\\ThoiToan\\Desktop\\Workspace\\reference\\pytorch-retinanet\\images\\000000000030.jpg")
img = img.to(device)

with torch.no_grad():
    classification, regression, anchors = model(img, return_raw=True)

print(regression)
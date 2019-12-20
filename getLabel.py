import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_path = "./inception-2015-12-05/cropped_panda.jpg"
image_path = "./outcwcat.jpg"
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
img = Variable(torch.from_numpy(img).to(device).float())

model = models.alexnet(pretrained=True).to(device).eval()
label = np.argmax(model(img).data.cpu().numpy())
print("label={}".format(label))

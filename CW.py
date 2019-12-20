# CW
import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import cv2
from tools import show_images_diff
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_path = "./inception-2015-12-05/cropped_panda.jpg"
image_path = "./1.jpg"
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)

k = 40
# model = "./inception-2015-12-05/classify_image_graph_def.pb"
model = models.alexnet(pretrained=True).to(device).eval()
boxmin = -3.0
boxmax = 3.0
boxmul = (boxmax - boxmin) / 2
boxplus = (boxmax + boxmin) / 2
#
# # 类别数：ImageNet图像库1000个类别
num_lables = 1000
#
# # 攻击目标标签必须是独热编码
target_label = 185
talb = Variable(torch.from_numpy(np.eye(num_lables)[target_label]).to(device).float())

# # Adam最大迭代次数
max_iterations = 10
#
# # Adam的学习速率
learning_rate = 0.01
#
# # 二分查找最大次数
binary_search_steps = 10
#
# # c的初始值
initial_const = 1e2
confidence = initial_const
#
lower_bound = 0
c = initial_const
upper_bound = 1e10
# # 保存最佳的L2值，预测概率和对抗样本
o_bestl2 = 1e10
o_bestscore = -1
shape = (1, 3, 224, 224)
o_bestattack = [np.zeros(shape)]
#
# # 迭代+二分查找 定义需要训练的modifer和Adam优化器
for outer_step in range(binary_search_steps):
    print("o_bestl2={}  confidence={}".format(o_bestl2, confidence))
    #把原始图像转换成图像数据和扰动的形态
    timg = Variable(torch.from_numpy(np.arctanh((img - boxplus) / boxmul * 0.999999)).to(device).float())
    modifier = Variable(torch.zeros_like(timg).to(device).float())
    modifier.requires_grad = True  ####
    optimizer = torch.optim.Adam([modifier], lr=learning_rate)
    #
    for iteration in range(1, max_iterations + 1):
        optimizer.zero_grad()
        print(modifier + timg)
        print(torch.tanh(modifier + timg))
        newimg = torch.tanh(modifier + timg) * boxmul + boxplus

        output = model(newimg.float())

        loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)
        real = torch.max(output * talb)
        other = torch.max((1 - talb) * output)
        loss1 = other - real + k
        loss1 = torch.clamp(loss1, min=0)
        loss1 = confidence * loss1
        loss = loss1 + loss2

        loss.backward(retain_graph=True)
        optimizer.step()

        L2 = loss2
        sc = output.data.cpu().numpy()

        if iteration % (max_iterations // 10) == 0:
            print("iterations={} loss={} loss1={} loss2={}".format(iteration, loss, loss1, loss2))
        if (L2 < o_bestl2) and (np.argmax(sc) == target_label):
            print("attack success L2={} target_label={}".format(L2, target_label))
            o_bestl2 = L2
            o_bestscore = np.argmax(sc)
            o_bestattack = newimg.data.cpu().numpy()

    confidence_old = -1
    if (o_bestscore == target_label) and o_bestscore != -1:
        # 攻击成功 减小c
        upper_bound = min(upper_bound, confidence)
        if upper_bound < 1e9:
            print()
            confidence_old = confidence
            confidence = (lower_bound + upper_bound) / 2
    else:
        lower_bound = max(lower_bound, confidence)
        confidence_old = confidence
        if upper_bound < 1e9:
            confidence = (lower_bound + upper_bound) / 2
        else:
            confidence *= 10

    print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, confidence))

adv = o_bestattack[0]
print(adv.shape)
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)

show_images_diff(orig, 0, adv, 0)
plt.matplotlib.image.imsave('outcwcat.jpg', adv)

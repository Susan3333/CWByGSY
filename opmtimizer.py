import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tools import show_images_diff

adam_original_loss = []
sdg_original_loss = []
RMSprop_original_loss = []
epoch_range = []


def run_adam_opt():
    global adam_original_loss
    global epoch_range
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "./inception-2015-12-05/cropped_panda.jpg"
    # image_path = "./outcw.jpg"
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
    # print("predicted:",decode_predictions(resnet50(img).data.cpu().numpy,top=3)[0])

    img.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam([img], lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    epochs = 100
    target = 288
    target = Variable(torch.tensor([float(target)]).to(device).long())

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(img)
        loss = loss_func(output, target)
        adam_original_loss += [loss]
        epoch_range += [epoch]
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots()
    ax.plot(np.array(epoch_range), np.array(adam_original_loss), 'b--', label='Adam')
    # ax.plot(np.array(epoch_range), np.array(RMSprop_original_loss), 'b-', label='RMSprop')
    # ax.plot(np.array(epoch_range), np.array(sdg_original_loss), 'b:', label='SGD')

    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#FFFFFF')

    plt.xlabel('Iteration Step ')
    plt.ylabel('Loss')
    plt.show()

    adv = img.data.cpu().numpy()[0]
    print(adv.shape)
    adv = adv.transpose(1, 2, 0)
    adv = (adv * std) + mean
    adv = adv * 255.0
    # adv = adv[..., ::-1]  # RGB to BGR
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    show_images_diff(orig, 388, adv, target.data.cpu().numpy()[0])
    plt.matplotlib.image.imsave('outopt.jpg', adv)


run_adam_opt()

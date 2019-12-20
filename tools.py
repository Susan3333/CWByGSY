import numpy as np
import matplotlib.pyplot as plt


def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img

    l0 = np.where(difference != 0)[0].shape[0]
    l2 = np.linalg.norm(difference)
    # print(difference)
    print("l0={} l2={}".format(l0, l2))

    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5

    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # plt.savefig(adversarial_img)



import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import models
from torchvision import transforms

import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image


if __name__ == '__main__':
    model= models.resnet18(pretrained=True)
    model.eval()
    target_layers=[model.layer1[-1].conv2]
    print(model)

    origin_img = cv2.imread('./data_content/T2_ann_imgs_0501/B10_1_end_T2_1.png')
    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224)
    ])
    crop_img = trans(rgb_img)
    net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)


    canvas_img = (crop_img*255).byte().numpy().transpose(1, 2, 0)
    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)


    cam = pytorch_grad_cam.GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(net_input)
    grayscale_cam = grayscale_cam[0, :]

    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)

    
    
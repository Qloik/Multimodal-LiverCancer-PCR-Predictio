import torch
from utils.args import _get_args
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
from models.pr import pCRModel
from PIL import Image
import os

activation = None
gradients = None

def forward_hook(module, input, output):
    global activation
    activation = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def attach_hooks(target_layer):
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

def compute_gradcam(model, input_data, target_class, target_layer):
    global activation, gradients

    # 清除之前的梯度
    model.zero_grad()

    # 将钩子附加到目标层
    attach_hooks(target_layer)

    # 前向传播
    output = model(input_data)
    
    # 获取目标类别的输出
    target = output[0, target_class]

    # 反向传播
    target.backward()

    # 计算权重
    weights = torch.mean(gradients, dim=[0, 2, 3])

    # 计算 Grad-CAM
    gradcam = torch.zeros_like(activation[0])
    for i, w in enumerate(weights):
        gradcam += w * activation[0, i]

    # 应用 ReLU 激活函数
    gradcam = torch.relu(gradcam)

    # 将 Grad-CAM 调整到原始图像大小
    gradcam = gradcam.cpu().detach().numpy()
    gradcam = cv2.resize(gradcam, (224, 224))

    # 标准化
    gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam))

    return gradcam

def show_gradcam(image_path, gradcam, alpha=0.5):
    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 生成热图
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加热图和原始图像
    superimposed_img = heatmap * alpha + image
    superimposed_img = superimposed_img / np.max(superimposed_img)

    # 显示结果
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

def process_image(image_path, model, target_layer, target_class):
    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载和预处理图像
    image = Image.open(image_path)
    input_data = {
        't1_start': image_transform(image).unsqueeze(0),  # 添加 batch 维度
        # 添加其他特征
        't1_end': image_transform(image).unsqueeze(0),  # 示例：使用相同的图像作为占位符
        'bef_afp': torch.tensor([0.0]),  # 示例：输入特征的占位符
        'bef_dcp': torch.tensor([0.0]),  # 示例：输入特征的占位符
        'aft_afp': torch.tensor([0.0]),  # 示例：输入特征的占位符
        'aft_dcp': torch.tensor([0.0])   # 示例：输入特征的占位符
    }

    # 计算 Grad-CAM
    gradcam = compute_gradcam(model, input_data, target_class, target_layer)

    # 显示 Grad-CAM
    show_gradcam(image_path, gradcam)

def main():
    # 设置路径和参数
    args = _get_args() 
    model_path = './runs/04151044-fold1/best_accuracy/checkpoint.pth.tar'  # 替换为模型路径
    image_path = './data_content/T2_ann_imgs_0414/A1_1_end_T2_0.png'  # 替换为图像路径
    target_class = 0  # 指定目标类别
    
    # 加载模型
    checkpoint = torch.load(model_path)
    model = pCRModel(args)  # 根据您的需要初始化 pCRModel

    # 加载模型权重
    model.load_state_dict(checkpoint['model'])
    target_layer = model.encoder_t1_start[7]  # 指定目标层
    # 选择目标层
    target_layer = getattr(model, target_layer)  # 例如，'encoder_t1_start.layer4'

    # 处理图像并计算 Grad-CAM
    process_image(image_path, model, target_layer, target_class)

if __name__ == '__main__':
    main()


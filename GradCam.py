import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# 全局变量，存储激活值和梯度
activation = None
gradients = None

# 前向钩子函数
def forward_hook(module, input, output):
    global activation
    activation = output

# 反向钩子函数
def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# 附加钩子到目标层
def attach_hooks(target_layer):
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

# 计算 Grad-CAM 的函数
def compute_gradcam(model, input_data, target_class, target_layer):
    global activation, gradients

    # 清除梯度
    model.zero_grad()

    # 将钩子附加到目标层
    attach_hooks(target_layer)

    # 前向传播
    output = model(input_data)

    # 提取指定类别的目标输出
    target = output[0, target_class]

    # 反向传播
    target.backward()

    # 计算权重
    weights = torch.mean(gradients, dim=[0, 2, 3])

    # 计算加权和
    gradcam = torch.zeros_like(activation[0])
    for i, w in enumerate(weights):
        gradcam += w * activation[0, i]

    # 应用 ReLU 激活函数
    gradcam = torch.relu(gradcam)

    # 将 Grad-CAM 上采样至输入图像大小
    gradcam = cv2.resize(gradcam.cpu().detach().numpy(), (input_data.shape[3], input_data.shape[2]))

    # 标准化 Grad-CAM
    gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam))

    return gradcam

# 显示 Grad-CAM 的函数
def show_gradcam(image_path, gradcam, alpha=0.5):
    # 读取原始图像
    image = Image.open(image_path)
    image = np.array(image)

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


    # 获取输出文件路径
    output_path = os.path.join(output_folder, os.path.basename(image_path))

    # 保存图像
    cv2.imwrite(output_path, (superimposed_img * 255).astype(np.uint8))

def main():
    # 设置路径和参数
    model_path = './runs/04151044-fold1/best_accuracy/checkpoint.pth.tar'  # 替换为您模型的路径
    image_path = './data_content/T2_ann_imgs_0414/A1_1_end_T2_0.png'  # 替换为您输入图像的路径
    output_folder = './outputGradCAM-04151044'  # 替换为您想要的输出文件夹路径
    target_class = 0  # 目标类别，根据您的模型需要调整
    alpha = 0.5  # 热图的透明度

    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    print(checkpoint.keys())
    # 从检查点中加载 ResNet 模型
    model = models.resnet18(pretrained=False)
    model.load_state_dict(checkpoint['model'])

    # 选择目标层（根据模型结构选择）
    target_layer = model.layer4  # 根据您的模型进行调整

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载和预处理图像
    image = Image.open(image_path)
    input_data = transform(image).unsqueeze(0)  # 添加 batch 维度

    # 计算 Grad-CAM
    gradcam = compute_gradcam(model, input_data, target_class, target_layer)

    # 显示 Grad-CAM
    show_gradcam(image_path, gradcam, alpha)

if __name__ == '__main__':
    main()


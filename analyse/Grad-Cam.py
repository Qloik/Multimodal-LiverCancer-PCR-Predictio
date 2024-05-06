import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from ..models.pr import pCRModel
# 定义前向和后向钩子
activation = None
gradients = None

def forward_hook(module, input, output):
    global activation
    activation = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# 将钩子附加到目标层
def attach_hooks(target_layer):
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

# 计算Grad-CAM函数
def compute_gradcam(model, input_data, target_class, target_layer):
    global activation, gradients

    # 将钩子附加到目标层
    attach_hooks(target_layer)
    
    # 前向传播
    model.zero_grad()
    output = model(input_data)
    
    # 指定目标类别
    target = output[0, target_class]

    # 反向传播
    target.backward()

    # 计算权重
    weights = torch.mean(gradients, dim=[0, 2, 3])

    # 计算加权和
    gradcam = torch.zeros_like(activation[0])
    for i, w in enumerate(weights):
        gradcam += w * activation[0, i]

    # 应用ReLU激活函数
    gradcam = torch.relu(gradcam)

    # 上采样Grad-CAM至原始图像大小
    gradcam = gradcam.cpu().detach().numpy()
    gradcam = cv2.resize(gradcam, (224, 224))

    # 标准化
    gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam))

    return gradcam

# 显示Grad-CAM的函数
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

    # 获取输出文件路径
    output_path = os.path.join(output_folder, os.path.basename(image_path))

    # 保存图像
    cv2.imwrite(output_path, (superimposed_img * 255).astype(np.uint8))



# 处理文件夹中的所有图像
def process_folder(folder_path, model, target_layer, target_class, output_folder):
    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 获取图像路径
            image_path = os.path.join(folder_path, filename)

            # 加载和预处理图像
            image = Image.open(image_path)
            input_data = image_transform(image).unsqueeze(0)  # 添加batch维度

            # 计算Grad-CAM
            gradcam = compute_gradcam(model, input_data, target_class, target_layer)

            # 显示Grad-CAM
            show_gradcam(image_path, gradcam, output_folder)



def main():

    # 设置路径和参数
    model_path = '../runs/04151044-fold1/best_accuracy/checkpoint.pth.tar'  # 替换为您的模型文件路径

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')

    # 加载您的pCRModel
    model = pCRModel(args)  # 使用您的模型类和参数

    # 设置目标层（根据您的模型结构选择）
    target_layer = model.encoder_t1_start.layer4  # 根据您的模型调整

    # 指定目标类别
    target_class = 0  # 根据您的目标类别进行调整

    # 设置图像文件夹路径
    folder_path = '../data_content/T2_ann_imgs_0414'  # 替换为您的文件夹路径
    
    # 设置输出文件夹路径
    output_folder = './outputGradCAM-04151044'  # 替换为您想要的输出文件夹路径

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的图像并计算 Grad-CAM
    process_folder(folder_path, model, target_layer, target_class, output_folder)



if __name__ == '__main__':
    main()


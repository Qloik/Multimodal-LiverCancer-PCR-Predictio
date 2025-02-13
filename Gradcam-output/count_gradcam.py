import csv
import cv2
import numpy as np
import os


def calculate_red_yellow_ratio(image):
    # 将图像转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色和黄色的 HSV 范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([60, 255, 255])


    # 创建红色和黄色区域的遮罩
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 计算红色和黄色区域的像素数量
    red_area = cv2.countNonZero(red_mask)

    # 计算整个图像的像素数量
    total_area = image.shape[0] * image.shape[1]

    # 计算红色和黄色区域占整个图像的比例
    red_ratio = red_area / total_area

    return red_ratio

if __name__ == "__main__":
    # 指定图像文件夹路径
    folder_path = "./data_analyse/start/FP"

    # 遍历文件夹中的所有图像文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # 创建 CSV 文件并写入标题行
    with open('start_FP_red_ratio.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'RedRatio'])

        # 对每张图像执行操作并将结果写入 CSV 文件
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            red_yellow_ratio = calculate_red_yellow_ratio(image)
            writer.writerow([image_file, red_yellow_ratio*100])

    print("结果已保存到 red_yellow_ratio.csv 文件中。")


    

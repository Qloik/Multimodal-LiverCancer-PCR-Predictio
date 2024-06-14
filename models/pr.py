import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as  F
class pCRModel(nn.Module):
    def get_model(self, resnet):
        return nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                                resnet.layer1, resnet.layer2, resnet.layer3,resnet.layer4, resnet.avgpool)
    # def depthwise_convolution(self, feature_map_start, feature_map_end):
    #     # 检查输入是否处于批处理模式下，并相应地调整维度
    #     is_batched = len(feature_map_start.size()) == 5

    # # 获取特征图的尺寸
    #     if is_batched:
    #         batch_size, channels, _, _, _ = feature_map_start.size()
    #     else:
    #         batch_size,channels, _, _ = feature_map_start.size()

    #     # 对每个通道进行逐通道卷积操作
    #     output = []
    #     for d in range(channels):
    #         if is_batched:
    #             feature_map_start_channel = feature_map_start[:, d, :, :, :].unsqueeze(1)  # 扩展维度以匹配卷积操作要求
    #             feature_map_end_channel = feature_map_end[:, d, :, :, :].unsqueeze(1)      # 扩展维度以匹配卷积操作要求
    #         else:
    #             feature_map_start_channel = feature_map_start[:,d, :, :].unsqueeze(1)  # 扩展维度以匹配卷积操作要求
    #             feature_map_end_channel = feature_map_end[:,d, :, :].unsqueeze(1)     # 扩展维度以匹配卷积操作要求

    #         conv_result = F.conv2d(feature_map_start_channel, feature_map_end_channel)
    #         output.append(conv_result)

    #     # 将每个通道的卷积结果拼接成一维向量
    #     output = torch.cat(output, dim=1)

    #     return output



    def __init__(self, args):
        super(pCRModel, self).__init__()
        self.args = args
        resnet1 = models.resnet18(pretrained=True)
        self.encoder_t1_start = self.get_model(resnet1)
        resnet2 = models.resnet18(pretrained=True)
        self.encoder_t1_end = self.get_model(resnet2)
        resnet3 = models.resnet18(pretrained=True)
        self.encoder_t2_start = self.get_model(resnet3)
        resnet4 = models.resnet18(pretrained=True)
        self.encoder_t2_end = self.get_model(resnet4)
        #resnet5 = models.resnet18(pretrained=True)
        #self.encoder_t3_start = self.get_model(resnet5)
        #resnet6 = models.resnet18(pretrained=True)
        #self.encoder_t3_end = self.get_model(resnet6)
        
        self.in_feature = resnet1.fc.in_features
        # self.fc = nn.Linear(3 * self.in_feature + 4, 1)  # 全连接层输出为单一值
        # self.bn = nn.BatchNorm1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4 * self.in_feature + 4, 20),
            nn.ELU(),
            nn.Linear(20, 20),
            nn.ELU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        
        f_t1_start = self.encoder_t1_start(x['t1_start'])
        f_t1_end = self.encoder_t1_end(x['t1_end'])
        f_t2_start = self.encoder_t2_start(x['t2_start'])
        f_t2_end = self.encoder_t2_end(x['t2_end'])
        

        # print("Shape of f_t1_start:", f_t1_start.shape)
        # print("size of f_t1_start:", f_t1_start.size)
        # print("Shape of f_t1_end:", f_t1_end.shape)
        # # 执行深度卷积操作
        # f_dw_conv = self.depthwise_convolution(f_t1_start, f_t1_end)
        # print(f_dw_conv.shape)
        # f_dw_conv = f_dw_conv.squeeze(3).squeeze(2)

        # # 对整个特征图进行平均池化
        # # 这将返回一个形状为 [batch_size, channels, 1] 的张量
        # f_dw_conv_pooled = F.avg_pool2d(f_dw_conv, kernel_size=f_dw_conv.size(2))
        
        # # 可选：将形状为 [batch_size, channels, 1] 的张量展平为形状为 [batch_size, channels] 的张量
        # f_dw_conv_pooled = f_dw_conv_pooled.squeeze(2)
        # # f_dw_conv = F.avg_pool3d(f_dw_conv, kernel_size=1)
        # out_pred_score = self.fc(f_dw_conv_pooled)
        # out_pred_score = self.bn(out_pred_score)
        # print(out_pred_score)




       # 将两个模态的特征以及深度卷积结果拼接在一起
        f_con = torch.cat([
                           torch.flatten(f_t1_start, start_dim=1),
            
                           torch.flatten(f_t1_end, start_dim=1),
                           #torch.flatten(f_t1_start, start_dim=1),
                           torch.flatten(f_t2_start, start_dim=1),
                           torch.flatten(f_t2_end, start_dim=1),
                    
                           x['bef_afp'].unsqueeze(1), x['bef_dcp'].unsqueeze(1), x['aft_afp'].unsqueeze(1), x['aft_dcp'].unsqueeze(1)], dim=1)
        logits = self.classifier(f_con)
        return logits

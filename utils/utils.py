from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
import time
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_root_dir = '../data'

warnings.filterwarnings("ignore")

class EvelRecord:
    def __init__(self) -> None:
        self.cache = dict(best_valid_accuracy=0.0, best_epoch=-1, best_valid_balanced_accuracy=0.0, best_valid_loss=np.inf)

    def update(self, values):
        if values['accuracy'] > self.cache['best_valid_accuracy']:
            self.cache['best_valid_accuracy'] = values['accuracy']
            self.cache['best_epoch'] = values['epoch']
            self.cache['best_valid_balanced_accuracy'] = values['balanced_accuracy']
            # self.cache['best_valid_loss'] = values['loss']
            return True
        return False
    
    def getitem(self, name):
        try:
            return self.cache[name]
        except:
            raise ValueError('Name is not included in the EvelRecord!')

def save_bag_result(results, logdir):
    if 'specificity' in results[0]:
        columns = ["accuracy", "precision", "recall", "f1", "specificity", "sensitivity"]
        results_df = pd.DataFrame(columns=columns)
        for dic in results:
            row_df = pd.DataFrame([[dic['accuracy'], dic['precision'], dic['recall'], dic['f1'], dic['specificity'], dic['sensitivity']]], columns=columns)
            results_df = pd.concat([results_df, row_df])
        results_df.to_excel(os.path.join(logdir, 'evaluation.xlsx'))
    else:
        columns = ["accuracy", "precision", "recall", "f1"]
        results_df = pd.DataFrame(columns=columns)
        for dic in results:
            row_df = pd.DataFrame([[dic['accuracy'], dic['precision'], dic['recall'], dic['f1']]], columns=columns)
            results_df = pd.concat([results_df, row_df])
        results_df.to_excel(os.path.join(logdir, 'evaluation.xlsx'))


def get_ce_gt(gt, kind):
    ce_gt = gt.detach().cpu()
    if kind == 'M0':
        ce_gt = (ce_gt != 0).long()
    elif kind == 'M2':
        ce_gt = (ce_gt == 2).long()
    ce_gt = ce_gt.cuda()
    return ce_gt

def save_confusion_matric(args, y, y_pred):
    C = confusion_matrix(y, y_pred, labels=[0, 1, 2])  # 可将'1'等替换成自己的类别，如'cat'。

    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=18)

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.savefig(os.path.join(args.logdir, 'confusion_matric.png'))

class TreeLoss:
    def __init__(self, args):
        self.args = args
        self.CELoss = torch.nn.CrossEntropyLoss()
    def get_tree_loss(self, logits_TF, logits_M1M2, gt):
        print(logits_TF, logits_M1M2, gt)
        loss = self.CELoss()

def save_attention_map_all(maps, filename, imgs):
    for l in range(32):
        save_attention_map_MANetCT(maps, filename, imgs, l)

def save_attention_map_MANetCT(maps, filename, imgs, layer):
    # layer = 15
    
    x1 = 0.85
    filename = filename[0]
    folder_name = f'visual/all_seq_fold1/{filename}'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(os.path.join(folder_name, 'T1')):
        os.mkdir(os.path.join(folder_name, 'T1'))
    if not os.path.exists(os.path.join(folder_name, 'T1D')):
        os.mkdir(os.path.join(folder_name, 'T1D'))
    # if filename not in {'42M1', '21M2', '28M2', '11M2', '42M2'}: 
    #     return
    def save_single_amap(mod):
        attention_map = F.interpolate(maps[mod][0:1, layer:layer + 1, :, :], size=224,
                                    mode='bilinear').squeeze().detach().cpu().numpy()
    # attention_map_threshold = attention_map_1 > x1
        img = np.load(os.path.join(f'{dataset_root_dir}/crop_data_224', f'{filename}_{mod}.npy'), allow_pickle=True)[0]
        # print(img.shape)
        # img = np.swapaxes(img, 0, 1)
        # img = np.swapaxes(img, 1, 2)
        # print(filename)
        # plt.imshow(img)
        # plt.show()
        # assert 0
        
        G_recon = (img) * 255.0
        G_recon = np.array(G_recon, dtype='uint8')
        G_recon = cv2.cvtColor(G_recon, cv2.COLOR_BGR2RGB)
        G_recon = G_recon[:, :, ::-1]
        if mod == 'T1':
            normed_mask = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
        else:
            normed_mask = 1 - (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
        threshold = (np.max(normed_mask) - np.min(normed_mask)) * 0.6 + np.min(normed_mask)
        # attention_map[attention_map < threshold] = 0 
        # normed_mask[normed_mask < threshold] = 0 
        # attention_map[attention_map < 1] = 0 
        normed_mask = np.uint8(255 * normed_mask)
        # print(normed_mask.max(), normed_mask.min())
        normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_RAINBOW)
        normed_mask = cv2.addWeighted(G_recon, 1, normed_mask, 0.2, 0)
        cv2.imwrite(f'{folder_name}/{mod}/{layer}.png', normed_mask)
        # plt.imshow(normed_mask)
        # plt.show()

    save_single_amap('T1')
    save_single_amap('T1D')

# def save_attention_map_MANetCT(maps, filename, imgs, layer):
#     # layer = 15
#     folder_name = f'visual/fold4-{layer}'
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#     x1 = 0.85
#     filename = filename[0]
#     # if filename not in {'42M1', '21M2', '28M2', '11M2', '42M2'}: 
#     #     return
#     def save_single_amap(mod):
#         attention_map = F.interpolate(maps[mod][0:1, layer:layer + 1, :, :], size=224,
#                                     mode='bilinear').squeeze().detach().cpu().numpy()
#     # attention_map_threshold = attention_map_1 > x1
#         img = np.load(os.path.join(f'{dataset_root_dir}/crop_data_224', f'{filename}_{mod}.npy'), allow_pickle=True)[0]
#         # print(img.shape)
#         # img = np.swapaxes(img, 0, 1)
#         # img = np.swapaxes(img, 1, 2)
#         # print(filename)
#         # plt.imshow(img)
#         # plt.show()
#         # assert 0
        
#         G_recon = (img) * 255.0
#         G_recon = np.array(G_recon, dtype='uint8')
#         G_recon = cv2.cvtColor(G_recon, cv2.COLOR_BGR2RGB)
#         G_recon = G_recon[:, :, ::-1]
#         if mod == 'T1':
#             normed_mask = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
#         else:
#             normed_mask = 1 - (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
#         threshold = (np.max(normed_mask) - np.min(normed_mask)) * 0.6 + np.min(normed_mask)
#         # attention_map[attention_map < threshold] = 0 
#         # normed_mask[normed_mask < threshold] = 0 
#         # attention_map[attention_map < 1] = 0 
#         normed_mask = np.uint8(255 * normed_mask)
#         # print(normed_mask.max(), normed_mask.min())
#         normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_RAINBOW)
#         normed_mask = cv2.addWeighted(G_recon, 1, normed_mask, 0.2, 0)
#         cv2.imwrite(f'{folder_name}/{filename}_{mod}.png', normed_mask)
#         # plt.imshow(normed_mask)
#         # plt.show()

#     save_single_amap('T1')
#     save_single_amap('T1D')

def save_attention_map(maps, filename, img):
    folder_name = 'visial2'
    x1 = 0.85
    filename = filename[0]
    attention_map_1 = F.interpolate(maps[0:1, 0:1, :, :], size=224,
                                    mode='bilinear').squeeze().detach().cpu().numpy()
    # attention_map_threshold = attention_map_1 > x1
    img = img.squeeze().detach().cpu().numpy()
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(os.path.join(folder_name, filename + '.npy')):
        np.save(os.path.join(folder_name, filename + '_attention_map.npy'), attention_map_1)
        np.save(os.path.join(folder_name, filename + '_img.npy'), img)

def get_modality_head(m):
    if len(m) > 1:
        return str(len(m)) + 'm'
    else:
        return m[0]


def get_str_time(x):
    if x < 10:
        s = f'0{str(x)}'
    else:
        s = str(x)
    return s


def get_time_head():
    struct_time = time.localtime(time.time())
    time_head = get_str_time(struct_time.tm_mon) + get_str_time(struct_time.tm_mday) + get_str_time(
        struct_time.tm_hour) + get_str_time(struct_time.tm_min)
    return time_head


def sensitivity_score(P0G0, P0G1, P0G2, P1G0, P1G1, P1G2, P2G0, P2G1, P2G2):
    if (P0G0 + P1G0 + P2G0) != 0:
        accuracy0 = P0G0 / (P0G0 + P1G0 + P2G0)
    else:
        accuracy0 = 0.0
    if (P0G1 + P1G1 + P2G1) != 0:
        accuracy1 = P1G1 / (P0G1 + P1G1 + P2G1)
    else:
        accuracy1 = 0.0
    if (P0G2 + P1G2 + P2G2) != 0:
        accuracy2 = P2G2 / (P0G2 + P1G2 + P2G2)
    else:
        accuracy2 = 0.0

    return accuracy0, accuracy1, accuracy2

def evaluate_prediction_entry(y, y_pred, is_multi = False):
    if np.sum(y_pred == 2) != 0 or is_multi:
        return evaluate_prediction_multiclass(y, y_pred)
    else:
        return evaluate_prediction(y, y_pred)

def evaluate_prediction_multiclass(y, y_pred):
    P0G0 = np.sum((y_pred == 0) & (y == 0))
    P0G1 = np.sum((y_pred == 0) & (y == 1))
    P0G2 = np.sum((y_pred == 0) & (y == 2))
    P1G0 = np.sum((y_pred == 1) & (y == 0))
    P1G1 = np.sum((y_pred == 1) & (y == 1))
    P1G2 = np.sum((y_pred == 1) & (y == 2))
    P2G0 = np.sum((y_pred == 2) & (y == 0))
    P2G1 = np.sum((y_pred == 2) & (y == 1))
    P2G2 = np.sum((y_pred == 2) & (y == 2))
    accuracy = (P0G0 + P1G1 + P2G2) / float(len(y))

    accuracy_0, accuracy_1, accuracy_2 = sensitivity_score(P0G0, P0G1, P0G2, P1G0, P1G1, P1G2, P2G0, P2G1, P2G2)
    balanced_accuracy = (accuracy_0 + accuracy_1 + accuracy_2) / 3.0

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    confusion_matric = dict(P0G0=int(P0G0), P0G1=int(P0G1), P0G2=int(P0G2), P1G0=int(P1G0), P1G1=int(P1G1), P1G2=int(P1G2), P2G0=int(P2G0), P2G1=int(P2G1), P2G2=int(P2G2))
    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'accuracy_0': accuracy_0,
               'accuracy_1': accuracy_1,
               'accuracy_2': accuracy_2,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               }
    return results


def evaluate_prediction(y, y_pred):
    """
    Evaluates different metrics based on the list of true labels and predicted labels.

    Args:
        y: (list) true labels
        y_pred: (list) corresponding predictions

    Returns:
        (dict) ensemble of metrics
    """
    true_positive = np.sum((y_pred == 1) & (y == 1))
    true_negative = np.sum((y_pred == 0) & (y == 0))
    false_positive = np.sum((y_pred == 1) & (y == 0))
    false_negative = np.sum((y_pred == 0) & (y == 1))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'precision': precision,
               'recall': recall,
               'ppv': ppv,
               'npv': npv,
               'f1': f1,
               }

    return results

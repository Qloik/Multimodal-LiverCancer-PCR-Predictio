from torch.utils import data
import numpy as np
import os
import re
import json
import logging
from torchvision import transforms
from PIL import Image

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

class pCR_Dataset(data.Dataset):
    def __init__(self, args, base_dir, transform=None, fold_type='folds', istrain='train'):
        self.args = args
        self.istrain = istrain
        self.base_dir = base_dir
        self.transform = transforms.Compose([ 
                    transforms.Resize((224, 224)),
                    transforms.AutoAugment(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.adapt_modality = self.args.modality
        self.fold_type = fold_type
        self._get_data_path()

    def _get_data_path(self):
        # with open(f'./data/data_fold_0308.json', 'r') as inf:
        with open(f'./data/data_fold_0423.json', 'r') as inf:
        # with open(f'../data/data_fold.json', 'r') as inf:
            data_dic = json.load(inf)
        with open(f'./data/clinic_data.json', 'r') as inf:
            clinic_data_dic = json.load(inf)
        data_dic = data_dic[self.istrain][str(self.args.fold)]
        self.gt_labels = list()
        self.filenames = list()
        self.subjects_start = list()
        self.subjects_end = list()
        self.clinic_data = clinic_data_dic
        images = os.listdir(self.base_dir)
        # a = 0
        for image in images:
            # a += 1
            head = image[:image.find('_')]
            if head in data_dic and 'start' in image and any(modality in image for modality in self.args.modality):
                if head == 'A23':
                    continue
                # if head == 'A96':
                #    continue
                end_filename = re.sub('start', 'end', image)
                self.filenames.append(head)
                self.subjects_start.append(os.path.join(self.base_dir, image))
                self.subjects_end.append(os.path.join(self.base_dir, end_filename))
                label = image.split('_')[1]
                self.gt_labels.append(label)
            
                # if label == '1' and self.istrain == 'train' and os.path.exists(os.path.join(f'/media/hdd1/ilab/lxy_media/2024MVI/data_content/pCR_mixup/fold{self.args.fold}', image)) and os.path.exists(os.path.join(f'/media/hdd1/ilab/lxy_media/2024MVI/data_content/pCR_mixup/fold{self.args.fold}', end_filename)):
                #     self.filenames.append(head)
                #     self.subjects_start.append(os.path.join(f'/media/hdd1/ilab/lxy_media/2024MVI/data_content/pCR_mixup/fold{self.args.fold}', image))
                #     self.subjects_end.append(os.path.join(f'/media/hdd1/ilab/lxy_media/2024MVI/data_content/pCR_mixup/fold{self.args.fold}', end_filename))
                #     self.gt_labels.append(label)


    def __getitem__(self, index):
        data = dict()
        head = self.filenames[index]
        data['start_filename'] = self.subjects_start[index]
        data['end_filename'] = self.subjects_end[index]
        data['filename'] = self.subjects_start[index]
        data['gt'] = np.array(self.gt_labels[index]).astype(np.int64)
        data['ouput']=0
        data['bef_afp'] = self.clinic_data[head]['bef_afp']
        data['bef_dcp'] = self.clinic_data[head]['bef_dcp']
        data['aft_afp'] = self.clinic_data[head]['aft_afp']
        data['aft_dcp'] = self.clinic_data[head]['aft_dcp']

        # 读取图像数据
        im_start = Image.open(data['start_filename'])
        im_end = Image.open(data['end_filename'])

         # 应用数据转换
        if self.istrain == 'test':
            im_start = self.transform_test(im_start)
            im_end = self.transform_test(im_end)
            
        else:
            im_start = self.transform(im_start)
            im_end = self.transform(im_end)

        # 将数据放入字典
        data['t1_start'] = im_start if 'T2' in data['start_filename'] else im_end  # 根据文件名判断模态
        data['t1_end'] = im_end if 'T2' in data['end_filename'] else im_start  # 根据文件名判断模态
     
        # data['t2_start'] = im_start if 'T2' in data['start_filename'] else im_end  # 根据文件名判断模态
        # data['t2_end'] = im_end if 'T2' in data['end_filename'] else im_start  # 根据文件名判断模态
        # data['t3_start'] = im_start if 'T3' in data['start_filename'] else im_end  # 根据文件名判断模态
        # data['t3_end'] = im_end if 'T3' in data['end_filename'] else im_start  # 根据文件名判断模态

        
        return data

    def __len__(self):
        return len(self.gt_labels)



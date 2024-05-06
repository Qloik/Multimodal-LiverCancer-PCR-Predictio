import os
import argparse
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.nn.functional import one_hot

from utils.utils import evaluate_prediction_entry
from utils.utils import save_confusion_matric, get_ce_gt
import torch.nn.functional as F

class Basic_Trainer():
    def __init__(self, args, criterion=None, train_loader=None, val_loader=None, writer=None, logger=None) -> None:
        self.args = args
        self.criterion = criterion
        self.val_loader = val_loader
        self.writer = writer
        self.logger = logger
        self.train_loader = train_loader
        self.columns = ["filename", "true_label", "predicted_label"]

    def train_log_scale(self, label_s, label_l, num, epoch):
        self.writer.add_scalar(label_s, num, epoch) 
        self.logger.info(f'{label_l}: {num}')

    def evaluate(self, results_df):
        results_df.reset_index(inplace=True, drop=True)
        results_df.sort_values(by=['true_label'], ascending=True, inplace=True)
        results = evaluate_prediction_entry(results_df.true_label.values.astype(int),results_df.predicted_label.values.astype(int))
        return results
    
    def get_row_df(self, filenames, gt, predicted):
        columns = self.columns
        row = []
        for idx, sub in enumerate(filenames):
            row.append([sub, gt[idx].item(), predicted[idx].item()])
        row_df = pd.DataFrame(row, columns=columns)
        return row_df
    
    def get_loss(self, logits, gt):
        if self.args.loss == 'focal_loss':
            onehot_target = one_hot(gt.long(), num_classes=self.args.class_num)
            loss = self.criterion(logits, onehot_target.float(), alpha=self.args.alpha, gamma=self.args.gamma, reduction='mean')
        else:
            loss = self.criterion(logits, gt.long())
        return loss

    def train(self, epoch, model, optimizer):
        self.logger.info('------- training -------')
        self.logger.info(f'Epoch: {epoch}')
        model.train()

        loss_list = []
        R_list = []
        num_processed = 0
        num_train = len(self.train_loader.dataset)
        results_df = pd.DataFrame(columns=self.columns)
        for batch_idx, data in enumerate(self.train_loader):

            model_input = dict()
            for m in self.args.modality:
                model_input[m] = data[m].cuda().float()
            gt = data['gt'].cuda()
            ce_gt = get_ce_gt(gt, self.args.kind)

            logits = model(model_input)
            loss = self.get_loss(logits, ce_gt)
            _, predicted = torch.max(logits.data, 1)      

            # Total loss
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            row_df = self.get_row_df(gt, predicted)
            results_df = pd.concat([results_df, row_df])
            
            # logging
            num_processed += len(gt)

        self.train_log_scale('train/total_loss', 'training loss', float(np.mean(loss_list)), epoch)

        results_train = self.evaluate(results_df)

        self.train_log_scale('train/accuracy', 'train accuracy', results_train['accuracy'], epoch)
        return model, np.mean(loss_list), results_train['accuracy']

    def validate(self, epoch, model):
        self.logger.info('------- validing -------')
        model.eval()
        results_df = pd.DataFrame(columns=self.columns)
        loss_list = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_loader)):

                model_input = dict()
                for m in self.args.modality:
                    model_input[m] = data[m].cuda().float()
                gt = data['gt'].cuda()
                ce_gt = get_ce_gt(gt, self.args.kind)

                logits = model(model_input)

                loss = self.get_loss(logits, ce_gt)
                loss_list.append(loss.item())
                _, predicted = torch.max(logits.data, 1)

                row_df = self.get_row_df(data['filename'], gt, predicted)
                results_df = pd.concat([results_df, row_df])

            results_valid = self.evaluate(results_df)
            mean_val_loss = np.mean(loss_list)
            results_valid['loss'] = mean_val_loss
            
            # logging
            self.train_log_scale('val/val_mean_loss', 'mean_val_loss', mean_val_loss, epoch)
            self.train_log_scale('val/accuracy', 'accuracy', results_valid['accuracy'], epoch)

            self.logger.info(f'gt_label: {results_df.true_label.values.astype(int)}')
            self.logger.info(f'pred_res: {results_df.predicted_label.values.astype(int)}')
            self.logger.info(f'epoch: {epoch}')
        return results_df, results_valid

    
    
    def test(self, model, test_loader):
        model.eval()
        results_df = pd.DataFrame(columns=self.columns)

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader)):

                model_input = dict()
                for m in self.args.modality:
                    model_input[m] = data[m].cuda().float()
                gt = data['gt'].cuda()
                
                logits = model(model_input)
                _, predicted = torch.max(logits.data, 1)
                score = F.softmax(logits, dim=1).cpu().detach().numpy()
                # print(score)

                row_df = self.get_row_df(data['filename'], gt, predicted)
                results_df = pd.concat([results_df, row_df])

            results_test = self.evaluate(results_df)
            if self.args.class_num == 3:
                save_confusion_matric(self.args, results_df.true_label.values.astype(int),
                                        results_df.predicted_label.values.astype(int))
            results_df.reset_index(inplace=True, drop=True)
            save_dir = os.path.join('/'.join(self.args.logdir.split('/')[:-1]), self.args.resume.split('/')[-1] + '_fold_{}.csv'.format(self.args.fold))
            self.logger.info('--- saving result in {} ---'.format(save_dir))
            results_df.to_csv(save_dir)
            
            # logging
            if self.args.class_num == 2:
                self.logger.info('gt_label: {}'.format(results_df.true_label.values.astype(int)))
                self.logger.info('pred_res: {}'.format(results_df.predicted_label.values.astype(int)))
                self.logger.info('test accuracy: {:.4f} \t f1 score: {:.4f}'.format(results_test["accuracy"], results_test["f1"]))
                self.logger.info('test precision: {:.4f} \t test recall: {:.4f}'.format(results_test["precision"], results_test["recall"]))
                self.logger.info('test sensitivity: {:.4f} \t test specificity: {:.4f}'.format(results_test["sensitivity"], results_test["specificity"]))
            else:
                self.logger.info('gt_label: {}'.format(results_df.true_label.values.astype(int)))
                self.logger.info('pred_res: {}'.format(results_df.predicted_label.values.astype(int)))
                self.logger.info(
                    'test accuracy: {:.4f} \t f1 score: {:.4f}'.format(results_test["accuracy"], results_test["f1"]))
                self.logger.info('test precision: {:.4f} \t test recall: {:.4f}'.format(results_test["precision"],results_test["recall"]))
        return results_test
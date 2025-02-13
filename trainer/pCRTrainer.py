import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .Basic import *

class pCR_Trainer(Basic_Trainer):
    def __init__(self, args, criterion=None, train_loader=None, val_loader=None, writer=None, logger=None) -> None:
        self.ce_loss1 = torch.nn.CrossEntropyLoss()
        self.ce_loss2 = torch.nn.CrossEntropyLoss()
        super().__init__(args, criterion, train_loader, val_loader, writer, logger)

    def get_loss(self, logits, gt):
        reweighted = False
        if reweighted:
            
            pos_index = (gt == torch.tensor(1)).nonzero().squeeze(1)
            pos_logits = torch.index_select(logits, 0, pos_index)
            pos_gt = torch.ones(pos_logits.size()[0], dtype=torch.int64).cuda()
            loss_pos = self.ce_loss1(pos_logits, pos_gt)

            neg_index = (gt == torch.tensor(0)).nonzero().squeeze(1)
            neg_logits = torch.index_select(logits, 0, neg_index)
            neg_gt = torch.zeros(neg_logits.size()[0], dtype=torch.int64).cuda()
            loss_neg = self.ce_loss2(neg_logits, neg_gt)

            pos = pos_gt.size()[0]
            neg = neg_gt.size()[0]

            # weight_pos = neg / (pos + neg)
            # weight_neg = pos / (pos + neg)
            weight_pos = (pos + neg) / pos if pos != 0 else 0
            weight_neg = (pos + neg) / neg if neg != 0 else 0
            if pos == 0:
                loss = loss_neg
            elif neg == 0:
                loss = loss_pos
            else:
                loss = weight_pos * loss_pos + weight_neg * loss_neg

            return loss
        else:
            return super().get_loss(logits, gt)

    def train(self, epoch, model, optimizer):
        self.logger.info('------- training -------')
        self.logger.info(f'Epoch: {epoch}')
        model.train()

        loss_list = []
        results_df = pd.DataFrame(columns=self.columns)
        for batch_idx, data in enumerate(self.train_loader):
            data['t1_start'] = data['t1_start'].cuda().float()
            data['t1_end'] = data['t1_end'].cuda().float()
            data['t2_start'] = data['t2_start'].cuda().float()
            data['t2_end'] = data['t2_end'].cuda().float()
        
        #    #data['t3_start'] = data['t3_start'].cuda().float()
        #    #data['t3_end'] = data['t3_end'].cuda().float()
        #    batch_size,channels, _, _ = data['t1_start'].size()

        #    output = []
        #    for d in range(channels):
        #        feature_map_start_channel = data['t1_start'][:,d, :, :].unsqueeze(1)  # 扩展维度以匹配卷积操作要求
        #        feature_map_end_channel = data['t1_end'][:,d, :, :].unsqueeze(1)     # 扩展维度以匹配卷积操作要求

        #        conv_result = F.conv2d(feature_map_start_channel, feature_map_end_channel)
        #        output.append(conv_result)
    
        #    # 将每个通道的卷积结果拼接成一维向量
        #    output = torch.cat(output, dim=1)
            
            #print(output.size)
            #print(output.shape)
        #    output=output.squeeze(3).squeeze(2)
        #    correlation_matrix = torch.corrcoef(output.T)
        #    mean_correlation = correlation_matrix.mean()
            
        #    print(mean_correlation.item())
            
            # add afp and dcp
            data['bef_afp'] = data['bef_afp'].cuda().float()
            data['bef_dcp'] = data['bef_dcp'].cuda().float()
            data['aft_afp'] = data['aft_afp'].cuda().float()
            data['aft_dcp'] = data['aft_dcp'].cuda().float()
            
            #data['output']=mean_correlation.item()
            #data['t1_start']=data['t1_start'] *  data['output']
            #data['t1_end']=data['t1_end'] *  data['output']

            


            gt = data['gt'].cuda()
            logits = model(data)
            
            # Loss
            loss = self.get_loss(logits, gt)
            # loss = self.criterion(logits, gt.long())
            _, predicted = torch.max(logits.data, 1) 

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            row_df = self.get_row_df(data['filename'], gt, predicted)
            results_df = pd.concat([results_df, row_df])
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
                data['t1_start'] = data['t1_start'].cuda().float()
                data['t1_end'] = data['t1_end'].cuda().float()
                # print(data['t1_start'].shape)
                
                data['t2_start'] = data['t2_start'].cuda().float()
                data['t2_end'] = data['t2_end'].cuda().float()
                
                #data['t3_start'] = data['t3_start'].cuda().float()
                #data['t3_end'] = data['t3_end'].cuda().float()
                # add afp and dcp
                data['bef_afp'] = data['bef_afp'].cuda().float()
                data['bef_dcp'] = data['bef_dcp'].cuda().float()
                data['aft_afp'] = data['aft_afp'].cuda().float()
                data['aft_dcp'] = data['aft_dcp'].cuda().float()

                gt = data['gt'].cuda()
                ce_gt = get_ce_gt(gt, self.args.kind)
                logits = model(data)
            
                soft = torch.nn.Softmax()(logits)
                _, predicted = torch.max(soft, 1)

                # loss = self.get_loss(logits, gt)
                # loss_list.append(loss.item())

                row_df = self.get_row_df(data['filename'], gt, predicted)
                results_df = pd.concat([results_df, row_df])

            results_valid = self.evaluate(results_df)
            results_df.reset_index(inplace=True, drop=True)
            # mean_val_loss = np.mean(loss_list)
            # results_valid['loss'] = mean_val_loss
            
            # logging
            
            self.logger.info(f'gt_label: {results_df.true_label.values.astype(int)}')
            self.logger.info(f'pred_res: {results_df.predicted_label.values.astype(int)}')
            self.logger.info(f'epoch: {epoch}')
            # self.train_log_scale('val/val_mean_loss', 'mean_val_loss', mean_val_loss, epoch)
            self.train_log_scale('val/accuracy', 'accuracy', results_valid['accuracy'], epoch)
            
        return results_df, results_valid

    
    
    def test(self, model, test_loader):
        model.eval()
        results_df = pd.DataFrame(columns=self.columns)

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader)):

                data['t1_start'] = data['t1_start'].cuda().float()
                data['t1_end'] = data['t1_end'].cuda().float()
                
                data['t2_start'] = data['t2_start'].cuda().float()
                data['t2_end'] = data['t2_end'].cuda().float()
                
                # data['t3_start'] = data['t3_start'].cuda().float()
                # data['t3_end'] = data['t3_end'].cuda().float()

                # add afp and dcp
                data['bef_afp'] = data['bef_afp'].cuda().float()
                data['bef_dcp'] = data['bef_dcp'].cuda().float()
                data['aft_afp'] = data['aft_afp'].cuda().float()
                data['aft_dcp'] = data['aft_dcp'].cuda().float()

                gt = data['gt'].cuda()
                
                logits = model(data)
                soft = torch.nn.Softmax()(logits)
                _, predicted = torch.max(soft, 1)

                row_df = self.get_row_df(data['filename'], gt, predicted)
                results_df = pd.concat([results_df, row_df])
            results_test = self.evaluate(results_df)
            results_df.reset_index(inplace=True, drop=True)
            save_dir = os.path.join('/'.join(self.args.logdir.split('/')[:-1]), self.args.resume.split('/')[-1] + '_fold_{}.csv'.format(self.args.fold))
            self.logger.info('--- saving result in {} ---'.format(save_dir))
            results_df.to_csv(save_dir)
            
            # logging
            self.logger.info('gt_label: {}'.format(results_df.true_label.values.astype(int)))
            self.logger.info('pred_res: {}'.format(results_df.predicted_label.values.astype(int)))
            self.logger.info('test accuracy: {:.4f} \t f1 score: {:.4f}'.format(results_test["accuracy"], results_test["f1"]))
            self.logger.info('test precision: {:.4f} \t test recall: {:.4f}'.format(results_test["precision"], results_test["recall"]))
            self.logger.info('test sensitivity: {:.4f} \t test specificity: {:.4f}'.format(results_test["sensitivity"], results_test["specificity"]))
            
        y_pred_res= [] # 创建一个空列表
        for value in results_df.predicted_label.values.astype(int):
            y_pred_res.append(value)# 将值添加到列表中
        y_pred=[]
        for num in y_pred_res:
            y_pred.append(str(num))
        
        y_true_res=[]
        for value in results_df.true_label.values.astype(int):
            y_true_res.append(value) # 将值添加到列表中
        y_true=[]
        for num in y_true_res:
            y_true.append(str(num))
        C = confusion_matrix(y_true, y_pred, labels=['0','1']) # 可将'1'等替换成自己的类别，如'cat'。
        plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
        TP_index = [i for i in range(len(y_pred)) if y_pred[i] == '1' and y_true[i] == '1']
        TN_index = [i for i in range(len(y_pred)) if y_pred[i] == '0' and y_true[i] == '0']
        FP_index = [i for i in range(len(y_pred)) if y_pred[i] == '1' and y_true[i] == '0']
        FN_index = [i for i in range(len(y_pred)) if y_pred[i] == '0' and y_true[i] == '1']
        #print("TP_index",TP_index)
        #print("TN_index",TN_index)
        #print("FP_index",FP_index)
        #print("FN_index",FN_index)
        
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()
        plt.savefig("./Gradcam-output/T2fold0_lr3_ResNet50.png")

        return results_test

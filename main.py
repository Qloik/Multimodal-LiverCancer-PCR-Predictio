import torch
import random
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from torch.utils import data
import torch
# 设置使用编号为1的GPU
torch.cuda.set_device(0)
from tensorboardX import SummaryWriter

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.logger import get_logger
from utils.io_tools import save_checkpoint, send_message
from utils.args import _get_args
from utils.utils import get_time_head

from trainer.pCRTrainer import pCR_Trainer

from models.pr import pCRModel

from dataset.MVI_dataset_pcr import pCR_Dataset

def get_args():
    args = _get_args()
    time_head = get_time_head()
    args.logdir = f'runs/{time_head}-fold{args.fold}'
    return args

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

def train_fold(args, logger, writer):

    logger.info(str(args))

    ##! building networks  
    logger.info('--- building networks ---')
    if args.arch == 'pCR':
        model = pCRModel(args)
    model = model.cuda()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('--- total parameters = {} ---'.format(n_params))

    ##! prepare dataset
    logger.info('--- loading dataset ---')

    train_loader = data.DataLoader(dataset=pCR_Dataset(args, base_dir=args.data_dir, transform=None, fold_type='folds', istrain='train'),
                                    batch_size=32,
                                    num_workers=args.num_workers,
                                    shuffle=False)
    val_loader = data.DataLoader(dataset=pCR_Dataset(args, base_dir=args.data_dir, transform=None, fold_type='folds', istrain='test'),
                                    batch_size=32,
                                    num_workers=args.num_workers,
                                    shuffle=False)

    ##! optimizer
    logger.info('--- configing optimizer ---')

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported Optimizer: {}'.format(args.optimizer))
    
    ##! loss
    logger.info('--- configing loss ---')

    if args.loss == 'ce_loss':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'focal_loss':
        criterion = sigmoid_focal_loss
    else:
        raise RuntimeError('unsupported loss: {}'.format(args.loss))
        
    ##! lr schedule
    logger.info('--- configing schedule ---')

    if args.lr_schedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.n_epochs)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    elif args.lr_schedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    else:
        raise RuntimeError('unsupported scheduler: {}'.format(args.lr_schedule))

    ##! start training 
    logger.info('--- start training ---')

    eval_record = EvelRecord()
    trainer = pCR_Trainer(args, criterion=criterion, train_loader=train_loader, val_loader=val_loader, writer=writer, logger=logger)
    logger.info(f'Train data: {len(train_loader.dataset)}, Test data: {len(val_loader)}')
    for epoch in range(args.n_epochs):
        model, train_loss, train_accuracy = trainer.train(epoch, model, optimizer)
        if (epoch + 1) % args.val_freq == 0:
            _, results_valid = trainer.validate(epoch, model)
        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr(), epoch)
    
        ## saving checkpoint 
        logger.info('--- saving checkpoint to {} ---'.format(args.logdir))

        results_valid['epoch'] = epoch
        isUpdate = eval_record.update(results_valid)
        if args.save_model == 1:
            save_checkpoint({'model': model.state_dict(), 'epoch': eval_record.getitem('best_epoch'), 'accuracy': eval_record.getitem('best_valid_accuracy')}, isUpdate, args.logdir
                        )
        logger.info('best accuracy: {} \t best epoch: {}'.format(eval_record.getitem('best_valid_accuracy'), eval_record.getitem('best_epoch')))
    writer.close()

if __name__ == '__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(args.logdir)
    logger = get_logger(args.logdir)
    train_fold(args, logger, writer)


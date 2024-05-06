import torch
import os

from torch.utils import data

from utils.logger import get_logger
from utils.io_tools import load_model
from utils.args import _get_args_test

from models.pr import pCRModel

from trainer.pCRTrainer import pCR_Trainer

from dataset.MVI_dataset_pcr import pCR_Dataset

def get_resume_path(resume_id):
    runs = os.listdir('./runs')
    possi_runs = list()
    for run in runs:
        if resume_id in run:
            possi_runs.append(run)
    assert len(possi_runs) == 1
    return f'./runs/' + possi_runs[0] + '/best_accuracy'

def get_args():
    args = _get_args_test()
    if args.resume[-1] == '/':
        args.resume = args.resume[:-1]
    args.resume = get_resume_path(args.resume)
    args.logdir = '/'.join(args.resume.split('/')[:-1])
    return args

def test_fold(args, logger):
    # init parameters
    
    logger.info(str(args))

    ## building networks  
    logger.info('--- building networks ---')
    model = pCRModel(args)
    trainer = pCR_Trainer(args, logger=logger)
    # model, _ = load_model(model, args.resume, args.gpu, filename='checkpoint.pth.tar')
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info('--- total parameters = {} ---'.format(n_params))
    # model = model.cuda()

    ## prepare dataset
    logger.info('--- loading dataset ---')
    test_loader = data.DataLoader(dataset=pCR_Dataset(args, base_dir=args.data_dir, transform=None, fold_type='folds', istrain='test'), batch_size=32, num_workers=args.num_workers, shuffle=False)
    logger.info('--- start testing ---')

    model, _ = load_model(model, args.resume, args.gpu, filename='checkpoint.pth.tar')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('--- total parameters = {} ---'.format(n_params))
    model = model.cuda()

    return trainer.test(model, test_loader)

    
if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    logger = get_logger(args.logdir)
    test_fold(args, logger)

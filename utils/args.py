import argparse

def base_arg():
    parser = argparse.ArgumentParser(description='MVI diagnosis')
    parser.add_argument('--dataset', default='pCR', choices=('pCR'))
    #parser.add_argument('--data_dir', default='T1_ann_imgs_0318')
    #parser.add_argument('--data_dir', default='T1V_ann_imgs_0501')
    parser.add_argument('--data_dir', default='T2_ann_imgs_0501')
    # parser.add_argument('--data_dir', default='T2+T1V_0501')
    # parser.add_argument('--data_dir', default='T1+T2+T1V')
    parser.add_argument('--fold', default=1, type=int)
    
    parser.add_argument('--arch', default="pCR", choices=('pCR'))
    parser.add_argument('--gpu', default=0, type=int, help='gpu ID')

    # Bottom setups are useless in this project for the moment
    parser.add_argument('--pretrain_arch', default='resnet34', choices=('vgg', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'))
    parser.add_argument('--modality', default=['T2'], nargs='+') #使用的模态
    parser.add_argument('--img_size', default=[3, 64, 64])
    parser.add_argument('--class_num', default=2, type=int, help='number of output class')
    parser.add_argument('--kind', default='M1M2', choices=('M0M1', 'M1M2'))
    parser.add_argument('--resume', default=None, type=str, help='use old folder to save result')
    parser.add_argument('--use_simclr', default=1, type=int)
    parser.add_argument('--transforms', default=1, type=int)
    return parser

def _get_args():
    parser = base_arg()
    parser.add_argument('--logdir', default='', type=str)
    parser.add_argument('--batch_size', default=32, type=int)# 32
    parser.add_argument('--num_workers', default=15, type=int)
    parser.add_argument('--channel', default=3, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='adamw', choices=('adam', 'adamw', 'sgd'), type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_schedule', default='cosine_anneal', choices=('step', 'warmup_cosine', 'cosine_anneal'), type=str)
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--loss', default='focal_loss', type=str, choices=('ce_loss', 'focal_loss'))
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--gamma', default=3, type=float)

    # training config
    parser.add_argument('--n_epochs', default=100, type=int)
    
    # val freq
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--save_model', default=1, type=int, choices=(0, 1))

    args = parser.parse_args()
    args = backward_args(args)
    return args

def backward_args(args):
    if args.dataset in {'MVI_crop_224', 'MVI_seg_224'}:
        args.img_size = [3, 224, 224]
    args.data_dir = './data_content/' + args.data_dir
    # args.data_dir = args.data_dir
    return args


def _get_args_test():
    parser = base_arg()
    # dataset
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    # The bottom setups are useless in this project for the moment
    # model
    parser.add_argument('--get_attention_map', default=0, type=int)
    # test config
    parser.add_argument('--mode', default='test', choices=('test', 'cam', 'tree'))
    parser.add_argument('--resume_forward')
    parser.add_argument('--resume_backward')
    args = parser.parse_args()
    args = backward_args(args)
    return args

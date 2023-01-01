import argparse
import os
import time
from torch.cuda.amp import autocast, GradScaler  # 混合精度
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from LSR import LSR
from model import *

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)
# load opts
parser.add_argument('--weights', type=str, required=False, default=None)
# save prefix
parser.add_argument('--save_prefix', type=str, required=True)
# dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--border', type=str2bool, required=True)
parser.add_argument('--mixup', type=str2bool, required=True)
parser.add_argument('--label_smooth', type=str2bool, required=True)
parser.add_argument('--se', type=str2bool, required=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'lrw':
    from utils import LRWDataset as Dataset
elif args.dataset == 'lrw1000':
    from utils import LRW1000_Dataset as Dataset
else:
    print(args.dataset)
    raise Exception('lrw or lrw1000')

avsr_model = AVSRModel(args).cuda()


def parallel_model(model):
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.AdamW(filter(lambda p: p.requires_grad, avsr_model.parameters()), lr=lr, weight_decay=1e-4, eps=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

if args.weights is not None:
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(avsr_model, weight.get('avsr_model'))

avsr_model = parallel_model(avsr_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if msg != '':
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


def test():
    with torch.no_grad():
        dataset = Dataset('val', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        print('start testing')
        v_acc = []
        total = 0
        for (i_iter, input) in enumerate(loader):
            avsr_model.eval()
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            audio = input.get('audio').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            with autocast():
                y_v = avsr_model(video, audio)

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if i_iter % 10 == 0:
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)
                print(msg)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        return acc, msg


def inference():
    print('My inference ...')
    res = []
    with torch.no_grad():
        dataset = Dataset('inference', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        print('start inferring ...')
        for (i_iter, input) in enumerate(loader):
            avsr_model.eval()
            video = input.get('video').cuda(non_blocking=True)
            audio = input.get('audio').cuda(non_blocking=True)
            # flipped_video = input.get('flipped_video').cuda(non_blocking=True)
            with autocast():  # 自动混合精度
                y_v = avsr_model(video, audio)
                # f_y_v = avsr_model(flipped_video, audio)
                # y_v = (y_v + f_y_v) / 2
            pred_ids = y_v.data.argmax(dim=-1).cpu().numpy().tolist()
            res.extend(pred_ids)
            print(pred_ids)

    with open('result.txt', 'w', encoding='utf-8') as fw:
        for x in res:
            fw.write(str(x) + '\n')
    print('DONE!!!')


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)


def token_cross_entropy(pred, tgt, lens):
    loss = torch.nn.functional.cross_entropy(pred.transpose(1, 2).contiguous(), tgt[:, None].expand(pred.shape[:2]).contiguous(), reduction='none')
    mask = torch.zeros(loss.size(0), loss.size(1)).float().to(pred.device)
    for i in range(lens.size(0)):
        L = min(mask.size(1), lens[i])
        mask[i, :L] = 1.0
    loss = (loss * mask).sum() / mask.sum()
    return loss


def train():
    dataset = Dataset('train', args)
    print('Start Training, Data Length:', len(dataset))
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    max_epoch = args.max_epoch
    tot_iter = 0
    best_acc = 0.0
    alpha = 0.2
    scaler = GradScaler()
    for epoch in range(max_epoch):
        lsr = LSR()
        for (i_iter, input) in enumerate(loader):
            tic = time.time()
            avsr_model.train()
            video = input.get('video').cuda(non_blocking=True)
            audio = input.get('audio').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()
            loss = {}
            if args.label_smooth:
                loss_fn = lsr
            else:
                loss_fn = nn.CrossEntropyLoss()
                # loss_fn = token_cross_entropy

            with autocast():
                if args.mixup:
                    # lambda_ = torch.distributions.beta.Beta(alpha, alpha)
                    lambda_ = np.random.beta(alpha, alpha)
                    index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                    mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                    label_a, label_b = label, label[index]
                    y_v = avsr_model(mix_video, audio)
                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                else:
                    y_v = avsr_model(video, audio)
                    loss_bp = loss_fn(y_v, label)

            loss['CE V'] = loss_bp
            optim_video.zero_grad()
            scaler.scale(loss_bp).backward()
            scaler.step(optim_video)
            scaler.update()
            toc = time.time()

            msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter, (toc - tic) * (len(loader) - i_iter) / 3600.0)
            for k, v in loss.items():
                msg += ',{}={:.5f}'.format(k, v)
            msg = msg + str(',lr=' + str(showLR(optim_video)))
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            print(msg)
            # if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):
            if (i_iter+1) % 1000 == 0:
                acc, msg = test()
                if acc > best_acc:
                    savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)
                    temp = os.path.split(savename)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    torch.save({'avsr_model': avsr_model.module.state_dict(),}, savename)
                if tot_iter != 0:
                    best_acc = max(acc, best_acc)
            tot_iter += 1
        scheduler.step()


if __name__ == '__main__':
    if args.test:
        print('===============================')
        inference()
        # acc, msg = test()
        # print(f'acc={acc}')
        exit()
    train()

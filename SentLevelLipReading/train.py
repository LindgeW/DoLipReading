import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from options import get_arg_parser
from logger import logger
from eval_metric import edit_dist
from data.grid import BOS, EOS, PAD
from transformers import get_linear_schedule_with_warmup


def get_dataset(opt, mode='train'):
    if opt.dataset == 'grid':
        from data.grid import GridSeq2Seq
        if mode == 'train':
            dataset_train = GridSeq2Seq(opt, phase='train')
            dataset_val = GridSeq2Seq(opt, phase='val')
            return dataset_train, dataset_val
        else:
            dataset_test = GridSeq2Seq(opt, phase='test')
            return dataset_test
    # elif opt.dataset == 'grid_raw':
    #     from data.grid import GridRaw
    #     dataset_train = GridRaw(opt, phase='train')
    #     dataset_val = GridRaw(opt, phase='val')
    #     dataset_test = GridRaw(opt, phase='test')
    else:
        raise


def get_model(opt):
    if opt.arch == 'Transformer':
        from archs.Transformer.model import Seq2Seq
        model = Seq2Seq(opt)
    elif opt.arc == 'RNN':
        from archs.RNN.model import Seq2Seq
        model = Seq2Seq(opt)
    elif opt.arc == 'CNN':
        from archs.CNN.model import Seq2Seq
        model = Seq2Seq(opt)
    else:
        model = None
    print(model)
    return model


def train(opt):
    train_set, val_set = get_dataset(opt, 'train')
    print(f'train set: {len(train_set)}, val set: {len(val_set)}')
    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers)

    model = get_model(opt).to(opt.device)
    param_nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {param_nums} trainable parameters')
    # model = nn.DataParallel(model)

    params = [{'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'weight_decay': opt.weight_decay, 'lr': opt.enc_lr},
        {'params': [p for p in model.decoder.parameters() if p.requires_grad],
         'weight_decay': opt.weight_decay, 'lr': opt.dec_lr}]
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    # 每过step_size个epoch，做一次lr更新
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_step, gamma=0.5)
    total_steps = len(train_loader) * opt.epochs
    warmup_linear_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    if opt.weights is not None:
        ckpt = torch.load(opt.weights, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('loading pre-trained weights ...')

    best_val_loss = 1e10
    for epoch in range(opt.epochs):
        model.train()
        ep_loss = 0.
        for step, batch in enumerate(train_loader):
            vid, align = batch[0], batch[1]
            vid_len, align_len = batch[2], batch[3]
            vid = vid.to(opt.device)
            align = align.to(opt.device)
            vid_len = vid_len.to(opt.device)
            align_len = align_len.to(opt.device)

            model.zero_grad()
            loss, _ = model(vid, align, vid_len, align_len)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            warmup_linear_scheduler.step()
            ep_loss += loss.item()

            if step % 5 == 0:
                logger.info('epoch: {} step: {}/{} train_loss: {:.4f}'.format(
                    epoch, (step+1), len(train_loader), loss.item()))
        logger.info('epoch loss: {}'.format(ep_loss / len(train_loader)))

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                vid, align = batch[0], batch[1]
                vid_len, align_len = batch[2], batch[3]
                vid = vid.to(opt.device)
                align = align.to(opt.device)
                vid_len = vid_len.to(opt.device)
                align_len = align_len.to(opt.device)
                loss, _ = model(vid, align, vid_len, align_len)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            logger.info('=========== val loss: {:.4f} ==========='.format(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('saved!!!')
            if opt.weights is None:
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('checkpoints', opt.dataset, 'best.ep{}.pt'.format(epoch)))
            else:
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('checkpoints', opt.dataset, 'continue.best.ep{}.pt'.format(epoch)))
        else:
            # lr_scheduler.step()  # adjust lr for each epoch
            pass


def evaluate(opt):
    test_set = get_dataset(opt, 'test')
    print(f'test set: {len(test_set)}')
    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.num_workers)

    model = get_model(opt).to(opt.device)
    print(model)
    checkpoint = torch.load(opt.load, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()

    wer_list = []
    cer_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            vid, align, align_txt = batch[0], batch[1], batch[2]
            vid = vid.to(opt.device)
            # res = model.greedy_decoding(vid, bos_id=BOS, eos_id=EOS)
            res = model.beam_search_decoding(vid, bos_id=BOS, eos_id=EOS)
            pred = list(map(lambda x: ''.join([test_set.idx_dict[i] for i in x if i != EOS and i != PAD]), res))
            print(pred, align_txt)
            wer_list.extend([edit_dist(p.split(' '), t.split(' ')) / len(t.split(' ')) for p, t in zip(pred, align_txt)])
            cer_list.extend([edit_dist(p, t) / len(t) for p, t in zip(pred, align_txt)])
    print('overall wer: {:.4f}, cer: {:.4f}'.format(np.mean(wer_list), np.mean(cer_list)))


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    opt = get_arg_parser()
    opt.device = torch.device('cuda', opt.cuda) if torch.cuda.is_available() and opt.cuda >= 0 else torch.device('cpu')
    print(opt.device)
    if opt.phase == 'test':
        evaluate(opt)
    else:
        train(opt)

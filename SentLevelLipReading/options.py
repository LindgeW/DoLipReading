import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='cuda device, default cpu')

    parser.add_argument('--dataset', type=str, default='grid')
    # parser.add_argument('--lrw_root', type=str, default='./dataset/lrw')
    # parser.add_argument('--grid_root', type=str, default=r'E:\GRID\LIP_160x80\lip')
    parser.add_argument('--video_root', type=str, default=r'E:\GRID\LIP_160x80\lip')
    parser.add_argument('--align_root', type=str, default=r'E:\GRID\LIP_160x80\align_txt')

    parser.add_argument('--arch', choices=['Transformer', 'RNN', 'CNN'], default='Transformer')
    parser.add_argument('--load', type=str, default='checkpoints/grid/best.ep0.pt')
    parser.add_argument('--weights', type=str, required=False, default=None)
    
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4, help='base learning rate')
    parser.add_argument('--enc_lr', type=float, default=5e-4)
    parser.add_argument('--dec_lr', type=float, default=5e-4)
    parser.add_argument('--loss_smooth', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--decay_step', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--out_channel', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--tgt_vocab_size', type=int, default=30)
    parser.add_argument('--tgt_pad_idx', type=int, default=0)
    parser.add_argument('--max_vid_len', type=int, default=75)
    parser.add_argument('--max_dec_len', type=int, default=80)

    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--enc_ffn_dim', type=int, default=512)
    parser.add_argument('--dec_ffn_dim', type=int, default=512)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_attn', type=float, default=0.1)
    parser.add_argument('--drop_embed', type=float, default=0.1)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

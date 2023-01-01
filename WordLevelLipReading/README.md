## Model Architecture

adapt from the [source code](https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains) of [Learn an Effective Lip Reading Model without Pains](https://arxiv.org/pdf/2011.07557.pdf) and integrate the audio features.

Frontend：3DCNN + SEResNet (for Video) + LogMelSpec + 1DCNN (for Audio)
Backend：BiGRU + Softmax
Training：Mixup + LabelSmoothing + CosineAnnealingLR
Testing：Test-Time Augmentation (horizontally flipped)

### Traing from scratch

python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=32 --num_workers=0 --max_epoch=20 --test=False --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True 


### Inference

python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=50 --num_workers=0 --max_epoch=20 --test=True --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True --weights=checkpoints/av-model.pt

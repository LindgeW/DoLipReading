<<<<<<< HEAD
## 模型结构：视听融合

改编自论文《Learn an Effective Lip Reading Model without Pains》中的唇语模型，并融入音频特征

前端：3DCNN + SEResNet (for Video) + LogMelSpec + 1DCNN (for Audio)
后端：BiGRU + Softmax
训练：Mixup + LabelSmoothing + CosineAnnealingLR
预测：Test-Time Augmentation (horizontally flipped)

# Traing from scratch
python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=32 --num_workers=0 --max_epoch=20 --test=False --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True 

# Inference
=======
## Word-level LipReading

Adapted from the [source code](https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains) of [Learn an Effective Lip Reading Model without Pains](https://arxiv.org/pdf/2011.07557.pdf), integrating the acoustic features.

- Frontend：3DCNN + SEResNet (for Video) + LogMelSpec + 1DCNN (for Audio)
- Backend：BiGRU + Softmax
- Training：Mixup + LabelSmoothing + CosineAnnealingLR
- Testing：Test-Time Augmentation (horizontally flipped)

Dataset: [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

### Training from scratch

python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=32 --num_workers=0 --max_epoch=20 --test=False --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True 


### Training from pretrained model

python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=32 --num_workers=0 --max_epoch=20 --test=False --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True --weights=checkpoints/lrw1000-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.48356.pt


### Inference

>>>>>>> 4acbca2a48c3d467951f97210de3b7207e91e79e
python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=50 --num_workers=0 --max_epoch=20 --test=True --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True --weights=checkpoints/av-model.pt


<<<<<<< HEAD

=======
### Resources
Audio-Visual LipReading：https://github.com/mpc001/end-to-end-lipreading/tree/master/audiovisual
>>>>>>> 4acbca2a48c3d467951f97210de3b7207e91e79e

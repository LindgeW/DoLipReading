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
python main_visual_audio.py --gpus=0 --lr=3e-4 --batch_size=50 --num_workers=0 --max_epoch=20 --test=True --save_prefix=checkpoints/retrained_lrw100
0/ --n_class=1000 --dataset=lrw1000 --border=False --mixup=True --label_smooth=True --se=True --weights=checkpoints/av-model.pt




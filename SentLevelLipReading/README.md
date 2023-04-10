### Sentence-level LipReading

Architecture: 3DCNN + Transformer + CTC/Attention + BeamSearch

Dataset: [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)


#### Training 

```
python train.py --cuda 0 --phase train --batch_size 32 --num_workers 2 [--weights checkpoints/grid/best.ep10.pt] --enc_layers 6 --dec_layers 6 --tgt_vocab_size 30 --grad_clip 1 --enc_lr 5e-4 --dec_lr 5e-4 --align_root /content/grid/align_txt --video_root /content/grid/lip
```


#### Testing

```
python train.py --cuda 0 --phase test --batch_size 32 --load checkpoints/grid/best.ep10.pt --align_root /content/grid/align_txt --video_root /content/grid/lip
```
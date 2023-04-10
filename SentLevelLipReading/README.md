### Sentence-level LipReading

Architecture: *3DCNN + Transformer + CTC/Attention + BeamSearch*

Dataset: [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)

#### Training 

```
python train.py --cuda 0 --phase train --batch_size 32 --num_workers 2 [--weights checkpoints/grid/best.ep10.pt] --enc_layers 6 --dec_layers 6 --tgt_vocab_size 30 --grad_clip 1 --enc_lr 5e-4 --dec_lr 5e-4 --align_root /content/grid/align_txt --video_root /content/grid/lip
```


#### Testing

```
python train.py --cuda 0 --phase test --batch_size 32 --load checkpoints/grid/best.ep10.pt --enc_layers 6 --dec_layers 6 --tgt_vocab_size 30 --align_root /content/grid/align_txt --video_root /content/grid/lip
```

#### Resources
+ https://github.com/arxrean/LipRead-seq2seq
+ https://github.com/bentrevett/pytorch-seq2seq
+ https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
+ https://github.com/liuxubo717/V-ACT/blob/main/tools/beam.py
+ https://github.com/haantran96/wavetransformer/blob/main/modules/beam.py
+ https://github.com/prajwalkr/vtp/blob/master/search.py

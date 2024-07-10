# WTConv
## Wavelet Convolutions for Large Receptive Fields. ECCV 2024.

*We will soon update this repository with a guide for training WTConvNeXt using 'timm'.*

[Model weights are available here](https://drive.google.com/drive/folders/1tiJjdEkYtw-2XKsQ61XzbMsncGXcWFmz?usp=sharing)

Requirements:
- Python 3.12
- timm 1.0.7
- PyWavelets 1.6.0

Running ImageNet-1K validation on WTConvNeXt-B:
```
python validate.py --data-dir IMAGENET_PATH --model wtconvnext_base --checkpoint WTConvNeXt_base_5_300e_ema.pth
```

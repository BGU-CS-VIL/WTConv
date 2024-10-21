# Semantic segmentation using MMSegmentation

## Usage

Follow MMSegmentation's [installation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation). 
The code was tested with mmsegmentation==2.2.0, python==3.9, pytorch==1.9.1 with cuda 11.1.

Also install [MMClassification](https://github.com/open-mmlab/mmclassification):

```shell
pip install mmpretrain>=1.0.0
```

Put the configurations and model files in the corresponding folder of MMSegmentation. Go to `mmseg/models/backbones/__init__.py` and add `from .wtconvnext import WTConvNeXt` as well as `WTConvNeXt` to the list of model names.

Note that the in the save file, the model keys are slightly different. You can use `translate_model_to_mmseg.py` script to change the pretrained save file to match the requirements.

Make sure you train the model using an effective batch size of 16, you can modify the batch size in the config (the config is tuned for 8 GPUs with batch size of 2).

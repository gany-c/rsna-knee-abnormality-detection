# ConvNeXt inference

Use `submit-knee-convnext-tiny-central-attention.ipynb` with the `gany24558/gc-rsna-knee-convnext/pyTorch/central-group-attention` model.

Build with `python build_inference_central_attention.py`. It reuses the base all-planes inference and decoder installer in `../resnet34/`. `resnet_inference_central_attention.py` retains its historical filename but implements ConvNeXt inference.

Matching training notebooks are in `../../training/convnext/`. Local folder names do not change Kaggle or Drive paths, model handles, or checkpoint identities.

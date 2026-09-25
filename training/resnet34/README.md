# ResNet34 training

Upload only `train-knee-resnet34.ipynb` to Kaggle. The Python sources are embedded.
Attach the complete normalized-training-data custom dataset and competition data.
Enable GPU and Internet. Run all cells.

The notebook makes a grouped 80/20 holdout, trains the backbone and classifier,
stops after three consecutive non-improving validation epochs, exports the best
checkpoint, and uploads it to `gany24558/gc-rsna-knee-resnet34/pyTorch/study-mil`.
The Kaggle Model is created when that real upload runs; no trained CNN has yet
been uploaded from this local development task.

See `resnet34-training-algorithm.md` for the design and terminology appendix.
This model needs a CNN-specific inference adapter; the DINO submission notebook
cannot load it. Preserve private session outputs for resume via `last.pt`.

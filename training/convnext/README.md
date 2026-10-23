# ConvNeXt training

- `rsna-convnext-tiny-central-attention.ipynb`: generated Kaggle training notebook.
- `rsna-convnext-tiny-central-attention-colab.ipynb`: generated Colab continuation notebook.
- `rsna_convnext_tiny_central_attention_colab.ipynb`: downloaded Colab run, including saved results.
- `rsna-rs34-central-attention.ipynb`: historical downloaded ConvNeXt run with its original filename.
- `last-kaggle-train-folder/`: original downloaded Kaggle output archive and checkpoint.

Build with `python build_notebook_central_attention.py`, then `python build_notebook_colab_continuation.py`. The training builder derives shared label/preprocessing code from `../resnet34/rsna-rs34-transfered-labels.ipynb`. Downloaded run notebooks are preserved, not regenerated.

Matching inference: `../../inference/convnext/submit-knee-convnext-tiny-central-attention.ipynb`.

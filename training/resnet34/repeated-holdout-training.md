# ResNet34: three repeated 25% holdouts

Upload `train-knee-resnet34-repeated-holdout.ipynb` to Kaggle. Existing notebooks remain unchanged. Four triplets and the original normalized image bundle are retained.

1. Attach the original normalized image dataset, RSNA competition dataset, and community label dataset (`labels_v1_blend.parquet`). Enable a GPU. Internet is needed to download initial ImageNet weights unless provided locally.
2. Set `MEMBER_INDEX=0`, leave `RESUME_TRAINING=False`, and run. The notebook fixes three holdout splits and the common 100-study selection set using seed 42.
3. If training stops at the 7.5-hour budget, save its outputs. Attach that output to the next run, keep the member and configuration unchanged, set `RESUME_TRAINING=True`, and provide the member's explicit `last.pt` path. Repeat until early stopping or the 35-epoch cap. Partial epochs are repeated on resumption.
4. Repeat for members 1 and 2, starting each fresh before resuming its own checkpoints. Do not use another member's checkpoint.
5. After each session with at least one completed validated epoch, its best checkpoint is exported and uploaded by default (`UPLOAD_MODEL=True`), including when the time budget stops training. Upload notes identify incomplete training and the stop reason. Resume from private `last.pt`; a later upload creates a new version of the same variation. Selection-set predictions are saved only once training finishes. Set `UPLOAD_MODEL=False` to opt out; if an upload fails, retry only the upload cell. Model variations are `comm-res-rh25-member-0`, `comm-res-rh25-member-1`, and `comm-res-rh25-member-2`.

All splits and private predictions are saved beneath `/kaggle/working/rsna-resnet34-repeated-holdout/<plan hash>/member-<index>/`. Preserve each member's outputs. Only the model package is uploaded; optimizer states, study IDs, targets, and predictions stay in private notebook outputs.

This is repeated holdout, not standard three-fold cross-validation. With 58 independent golden study groups, each member validates 15 studies and can train on 43 golden studies. Validation sets may overlap, and some golden studies are never validated. The notebook prints coverage and overlap. Never include predictions from a model trained on an example when calculating its held-out score. Checkpoint selection itself uses the golden holdout, so these remain development estimates.

The 100 selection studies are drawn from non-golden studies with usable labels and available images. They and their patient groups are excluded from every training member. Golden patient groups are ineligible for selection. With patient grouping, actual golden study counts may differ from 15; selection-related exclusions may also remove additional studies. The notebook reports actual counts. Patient separation cannot be guaranteed where only study IDs exist.

Selection predictions use fixed, smoothed community targets, not radiologist ground truth. No radiologist labels are currently available. Predictions support later comparison of individuals, pairs, or the three-member average; the notebook does not yet rank ensembles or create ensemble inference. Selection loss does not control early stopping. Findings without both official classes have unavailable AUROC and are reported.

Training uses one 7.5-hour budget per execution, followed by export and bounded validation/selection prediction passes. Train only one member per execution. Submission inference still has to fit the saved nine-hour competition limit; benchmark it before deploying multiple members.

## Local validation

Split tests cover deterministic selection, holdout sizes, study/patient exclusions, and invalid inputs. Notebook code cells compile. Full GPU training and export require Kaggle inputs and have not been run locally.

```sh
python3 -m unittest discover -s training/resnet34 -p test_repeated_holdout_splits.py -v
python3 training/resnet34/build_notebook_repeated_holdout.py
```

The builder reads the original four-triplet notebook as a runtime template, embeds the split helper, and writes only the new repeated-holdout notebook. Rebuilding clears that notebook's saved execution outputs.

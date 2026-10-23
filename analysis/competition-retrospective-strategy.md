# Retrospective Strategy

1. Partially labelled data - Check if there is a community given label set, check if there are public models available for pseudo-labelling
2. Teaming up with a Radiologist - It is hard, one finally worked after 3; it is easy to set up a labeling local website - Thanks to vibe coding.
3. Next time, if you have both of the above and the radiologist is available early on, this may be a good strategy:
   1. pseudo label first and select positive and negative examples for all labels
   2. get the radiologist to label these and add them to the training set as gold labels create models
   3. Then, iteratively add more 10-20 labelled samples using the radiologist every week across all labels and low confidence samples
4. Models - Dino vs Resnet vs ConvNext
   1. Resnet and Dino finish within the competition time, but ConvNext doesn't
   2. Resnet gives better scores than Dino
   3. While training if you see that some model massively expands the epoch time, it will also eat into the scoring or inference time. Think about inference times also, Don't just think about adding resources to training.
   4. Ensembles give slightly better results - Dino and Resnet ensemble, Resnet and Resnet ensembles i.e. with different validation and training sets.
5. Your own innovation
   1. The one that worked was placing more emphasis on the central slices, it caused a decent jump in the score.
6. Kaggle tricks and tips
   1. "Save and Run all" lets you run the training in the background, you don't have to babysit the notebook session for several hours.
   2. The output of a Notebook can be used as input to the same notebook or different; or you can export it to a dataset.
      1. This helps you stagger a long training run across 1-2 days. The results of previous epochs are stored as last.pt and best.pt; even the PATIENCE/Early Stopping related information can stored
   3. A model can also be stored as a dataset
   4. You can store Python utilities e.g. Dicom Encoders as a dataset. This will be useful in inference where there is no internet access.
   5. You can buy more compute by purchasing a subscription  in Google Colab. Buying pay as you go doesn't add to your Kaggle capacity.
   6. You can run your training directly in Colab; the datasets, models etc are accessed through Kaggle APIs

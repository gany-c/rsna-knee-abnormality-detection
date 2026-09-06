# RSNA Knee Abnormality Detection --- EDA Learnings

## Combined Learnings

1.  **Dataset hierarchy:**\
    **Study → Series → DICOM images/slices.** One MRI study contains
    several series, and each series contains multiple 2D image slices.

2.  **Slices are spatial, not temporal.** Images within a series
    represent neighboring positions through the knee, not different
    moments in time. Together, they approximate a 3D view/volume.

3.  **Different series are complementary views of the same knee.** They
    can differ by anatomical plane and MRI acquisition settings, making
    different structures/abnormalities more visible.

4.  **The 12 abnormality labels are study-level.** We ultimately predict
    12 probabilities for the **entire study**, rather than separately
    labeling individual slices or series.

5.  **Gold labels are extremely sparse.** Only **58 of 4,407 studies**
    have all 12 labels; the remaining **4,349 have none**. All targets
    therefore have only **1.32% gold-label coverage**.

6.  **Missing labels mean unknown, not negative.** They must therefore
    be **masked during training**, rather than filled with zero.

7.  **The 12 targets have different class balances.** Among the 58
    labeled studies, MCL is the rarest (**15.5% positive**) while
    Effusion is the most common (**60.3% positive**).

8.  **Some abnormalities co-occur.** Examples include
    Effusion/Synovitis, ACL/Contusion and compartment-specific
    OA/Meniscus relationships. However, correlations based on only 58
    studies are highly uncertain.

9.  **All 4,407 studies contain radiology reports.** The reports are
    particularly valuable because they could potentially be used to
    **derive/pseudo-label the 4,349 unlabeled studies**.

10. **Reports are unavailable at test time.** They should therefore be
    treated as **privileged training information**, not something the
    final inference pipeline depends upon.

11. **Reports are multilingual.** 4,301 contain Latin script and 220
    contain Cyrillic. Latin doesn't imply English specifically; actual
    languages haven't yet been identified.

12. **Report vocabulary is highly relevant to the targets.** Common
    terms include `meniscus`, `tear`, `ligament`, `intact`, `effusion`,
    `edema`, etc.

13. **Simple keyword matching isn't enough for label extraction.**
    `"ACL tear"` and `"ACL intact"` both contain ACL terminology but
    imply opposite labels. An NLP/LLM system must understand **negation,
    anatomy, uncertainty, context and multiple languages**.

14. **An LLM is a plausible pseudo-labeling approach.** It could
    classify each report/target as positive, negative, uncertain or not
    mentioned. Its performance should first be measured against the **58
    gold-labeled studies**.

15. **`train_series.csv` describes 24,371 MRI series.** A typical study
    contains about **5--6 series**: mean **5.53**, median **5**, with a
    range of **3--14**.

16. **Every training study contains all three anatomical planes:**
    **Sagittal, Coronal and Axial** are each present in 100% of studies,
    although a study may contain multiple series in the same plane.

17. **Acquisition protocol describes how a series was captured.** For
    this EDA, it is represented using **Anatomical Plane + Fluid
    Sensitive + Fat Suppression**.

18. **Fluid sensitivity and fat suppression perfectly coincide in this
    training data:** only `0/0` and `1/1` were observed. The competition
    explicitly warns that this relationship **shouldn't be assumed for
    unseen data**.

19. **DICOM files contain pixels plus metadata.** Metadata describes
    properties such as image dimensions, pixel spacing, slice thickness,
    bit depth, spatial information and transfer syntax.

20. **The DICOM audit deliberately samples rather than scanning \~570
    GB.** It selects **60 series** reproducibly and examines the header
    of a representative middle slice from each.

21. **The sampled series show substantial heterogeneity.** They contain
    **16--176 slices**, with a median of 30. Image dimensions, physical
    pixel spacing, slice thickness and bit depth also vary.

22. **Raw image dimensions aren't standardized.** Sampled images ranged
    roughly from **256×256 up to 1024×1280**, with 512×512 being
    typical.

23. **Physical resolution isn't standardized either.** Pixel spacing and
    slice thickness vary, meaning that two identically sized pixel
    arrays don't necessarily represent the same physical area/volume of
    the knee.

24. **Actual DICOM pixel decoding was tested on 12 representative middle
    slices.** All 12 decoded successfully, and the images visibly
    differed in resolution, anatomical view, contrast and content.

25. **Preprocessing therefore has multiple levels:**\
    **standardize/resample individual slices → handle variable slice
    counts within a series → handle variable series counts/protocols
    within a study.**

26. **The natural model hierarchy is therefore:**\
    **DICOM slices → series representation → combine series/planes →
    study representation → 12 predictions.**

27. **Validation must be study-level.** All slices, series and report
    information from one study must remain in the same fold; otherwise
    we'd introduce data leakage.

28. **Class imbalance should be handled per target.** Weighting,
    balanced sampling, focal/asymmetric losses, etc. can be compared
    using **per-target and macro AUC rather than accuracy**.

29. **We shouldn't assume test prevalence matches training prevalence.**
    The competition explicitly warns that abnormality prevalence can
    shift between training and evaluation datasets.

30. **Caching/precomputing will be important.** With \~570 GB of DICOM
    data, repeatedly decoding and preprocessing the entire raw corpus
    for every experiment would be impractical.

31. **A sensible first modeling experiment is three leakage-safe
    baselines on identical folds:**\
    **label-prior → TF-IDF report model → small 2.5D MRI model.**\
    This tells us how much signal comes from prevalence, text and images
    before investing in a complex architecture.

## Big Picture

> **4,407 studies → \~5--6 MRI series/study → multiple spatial DICOM
> slices/series → only 58 gold-labeled studies but reports for all 4,407
> → use reports to create additional supervision → standardize
> heterogeneous MRI data → aggregate slice → series → study information
> → predict 12 abnormalities.**

# RSNA Knee Abnormality Detection --- Notes

## 1. Multimodal Data

**Multimodal data** means using more than one type of information in a
machine-learning problem.

Examples of modalities include: - Images - Text - Audio/video - Tabular
or numerical data - Time-series data

For a knee MRI problem, a multimodal model could potentially combine MRI
images with radiology reports, clinical information, or acquisition
metadata.

------------------------------------------------------------------------

## 2. Knee Abnormality Labels

The competition predicts several abnormalities:

-   **ACL** --- Injury to the anterior cruciate ligament, an important
    ligament inside the knee that helps control forward movement and
    twisting.
-   **MCL** --- Injury to the medial collateral ligament, which
    stabilizes the inner side of the knee.
-   **Medial Meniscus** --- Tear of the shock-absorbing cartilage on the
    inner side of the knee.
-   **Lateral Meniscus** --- Tear of the shock-absorbing cartilage on
    the outer side of the knee.
-   **Medial OA** --- Osteoarthritis (wear and tear) affecting the inner
    part of the knee joint.
-   **Lateral OA** --- Osteoarthritis affecting the outer part of the
    knee joint.
-   **PF OA** --- Patellofemoral osteoarthritis, affecting the joint
    around/behind the kneecap.
-   **Effusion** --- Excess fluid inside the knee joint, often
    associated with injury or inflammation.
-   **Synovitis** --- Inflammation of the lining of the knee joint.
-   **Baker's cyst** --- A fluid-filled swelling behind the knee, often
    associated with other knee problems.
-   **Contusion** --- A bone bruise: injury inside a bone without an
    actual fracture.
-   **Fracture** --- A crack or break in a bone.

A useful grouping is:

  Category                    Labels
  --------------------------- -----------------------------------
  Ligaments                   ACL, MCL
  Menisci / shock absorbers   Medial Meniscus, Lateral Meniscus
  Osteoarthritis              Medial OA, Lateral OA, PF OA
  Fluid / inflammation        Effusion, Synovitis
  Fluid-filled cyst           Baker's cyst
  Bone injuries               Contusion, Fracture

------------------------------------------------------------------------

## 3. Study-Level Labels

A **study** is the complete MRI examination of a knee. It can contain
multiple MRI series and many individual images.

A **study-level label** gives a diagnosis for the entire MRI examination
rather than for a particular image.

For example:

``` text
Study 101
  ├── Series 1
  ├── Series 2
  └── Series 3

Study-level labels:
  ACL = 1
  MCL = 0
  Medial Meniscus = 1
  Fracture = 0
```

`ACL = 1` means the study is positive for an ACL injury. It does **not**
tell us which individual MRI slice shows the injury.

Therefore, study-level labels should not automatically be copied onto
every image in the study.

------------------------------------------------------------------------

## 4. Series-Level Acquisition Metadata

Within one MRI study there can be several **series**. Each series is a
set of images acquired using a particular MRI orientation or sequence.

The hierarchy is:

``` text
Study
  └── Series
        └── Individual MRI images/slices
```

**Series-level acquisition metadata** describes **how a particular MRI
series was acquired**, rather than the diagnosis.

Examples can include: - Anatomical plane: sagittal, coronal, axial - MRI
sequence/type - Fat suppression - Fluid sensitivity - Slice thickness -
Pixel spacing/resolution - Repetition time (TR) - Echo time (TE) - Other
scanner/acquisition parameters

In short:

  -----------------------------------------------------------------------
  Level                   Example                 Meaning
  ----------------------- ----------------------- -----------------------
  Study-level label       ACL = 1                 What abnormality is
                                                  present

  Series-level metadata   Sagittal,               How that series was
                          fat-suppressed          acquired

  Image data              MRI pixels              What the anatomy
                                                  actually looks like
  -----------------------------------------------------------------------

### Standard anatomical planes

A **standard plane** is a reference direction for taking slices through
the body. Imagine looking at the knee from the front, from the side, or
across the leg.

| Plane | Simple way to picture it | Divides the body into |
|---|---|---|
| **Coronal** | A view from the front | Front and back portions |
| **Sagittal** | A view from the side | Left and right portions |
| **Axial** | A cross-section across the leg, like cutting a loaf into slices | Upper and lower portions |

Actual MRI series can be **tilted** relative to these reference directions.
This is called an **oblique acquisition** and may help show a particular
structure. A tilted series is not automatically incorrect or unusable.

**Closest standard plane** means the reference plane most closely aligned
with the acquisition orientation recorded in the DICOM metadata. It is
not determined by how the image is rotated on the screen.

### What the preprocessing angle check means

The notebook compares the plane named in `train_series.csv` with the
orientation recorded inside the DICOM images. Think of a photo labeled
“front view” whose camera was actually positioned at an angle.

- **0° difference:** aligned with the named plane.
- **90° difference:** perpendicular to the named plane.
- **More than 35°:** exceeds the notebook's current configured cutoff.

The **35° cutoff is our preprocessing choice**, not a universal rule about
whether an MRI is valid. Exceeding it can reflect a strongly tilted
acquisition or inconsistent plane metadata; the angle alone does not
establish which explanation is correct.

When this check fails, the code records an error and skips that series.
Under the current complete-study requirement, its study is also excluded
from the training-ready manifest. **The source images are not deleted.**
The failed-series diagnostic cells display original slices and print the
measured angle, cutoff, and closest standard plane in red for review.

------------------------------------------------------------------------

## 5. Efficiency Score

The competition's efficiency metric is:

\[ Efficiency = `\frac{AUC}{Benchmark - \max AUC}`{=tex} +
`\frac{RuntimeSeconds}{32400}`{=tex} \]

The objective is to **minimize** this score.

### Why minimize?

Because normally:

\[ Benchmark - `\max `{=tex}AUC \< 0 \]

Therefore, increasing AUC makes the first term more negative, which
lowers the efficiency score.

At the same time, increasing runtime makes the second term larger, which
raises the score.

So the metric rewards:

-   **Higher predictive performance (AUC)**
-   **Lower inference/evaluation runtime**

### Why divide runtime by 32,400?

\[ 32400`\text{ seconds}`{=tex} = 9`\text{ hours}`{=tex} \]

The divisor normalizes runtime onto a manageable numerical scale. A
9-hour runtime contributes `1.0` to the runtime term, while a 1-hour
runtime contributes about `0.111`.

The practical implication is that a much larger or slower model is
worthwhile only if its improvement in predictive performance compensates
for its additional runtime.

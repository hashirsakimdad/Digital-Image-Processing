# Digital Image Processing

Lab code for the Digital Image Processing course (Fall 2026, PAF-IAST), by Hashir (Reg No B23F0435AI108).

| Lab | Topic | Code |
|-----|-------|------|
| Lab 1 | Image loading, properties, pixel access, saving (OpenCV, Pillow, Matplotlib) | `Lab Tasks/Lab task 1/labtask1.ipynb` |
| Lab 2 | Colour conversion, crop, resize, flip | `generate_lab_outputs.py` (section for Lab 2) |
| Lab 3 | Arithmetic operations, white-space count, bounding boxes | `Lab Tasks/Lab task 3/labtask3.ipynb` |
| Lab 4 | Two-image operations, blending, bitwise masks, change detection | `Lab Tasks/Lab task 4/labtask4.ipynb` |
| Lab 5 | Intensity transformations for retinal image enhancement | `Lab05_Intensity_Transformations.ipynb` |

`generate_lab_outputs.py` regenerates the Lab 1 to 4 output images from a single image file.

## Data

- Lab 3 uses a PKLot parking-lot sample (Almeida et al., 2015), a forest photo (Unsplash) and NumPy-generated images. See `Lab Tasks/Lab task 3/SOURCE.txt` and `dataset/SOURCE.txt`.
- Lab 4 uses the RESECT dataset (Case1). The image files are not included here because of their licence terms. Download them from the source listed in `Lab Tasks/Lab task 4/mri_dataset/SOURCE.txt`, then place them in that folder.
- The Lab 1 photo is a personal identity document and is not included.

## Requirements

Python 3 with `opencv-python`, `numpy`, `matplotlib` and `pillow`.

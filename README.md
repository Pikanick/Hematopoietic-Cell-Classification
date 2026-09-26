## ENSC 413 - Final Project Code 
### Getting Started
This small code base contains the training, testing, and prediction scripts for the classification of white blood cells using a convolution neural network. Before proceeding please be aware the [BMC-Dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) by Matek et al. is required for training and testing. However, for inference, you can run `python3 Model_Predict.py`.
The script will load a trained model and make a prediction on the file(s) contained within `./Sample_Images/`. For fun try changing line 11 in `Model_Predict.py` to attempt at classifying different cells. Finally, a Tesla EV image was added; try seeing what cell class our CNN thinks a Tesla Model S is.

### Project Layout
- `Models/` is a directory containing the saved final model (a retrained half-width MV3; see "Trained model" below).
- `Weights/` is a directory containing the MV3 weights.
- `Sample_Images/` is a directory of sample images for inferance. 
- `Utility_Functions` is a directory of different scripts for pre-processing,
    augmentation, and the creation of a data generator.
- `Model_Train.py` is a script for training the final proposed MV3 architecture in "Investigating CNN Viability and Performance within Hematopoietic Cell Classification".
- `Model_Test.py` was used to obtain test metrics for the final model comparison.
- `Mode_Predict.py` is a fun script to predict images contained within `Sample_Images`.


### Trained model (retrained, half-width)
The original weights from the paper were never uploaded to Git LFS, so the model was **retrained** from scratch. To make training feasible on free Colab GPUs, this version uses **half the conv widths** of the paper's MV3 (128/64/32 instead of 256/128/64, `MV3_WIDTH_DIV=2`), mixed-precision training, and 30 epochs (best checkpoint = epoch 18). It is therefore *not* identical to the model evaluated in the paper.

Test-set results (7,763 held-out images):

| Class | Precision | Recall |
|---|---|---|
| BLA | 0.895 | 0.806 |
| LYT | 0.927 | 0.875 |
| NGB | 0.717 | 0.586 |
| NGS | 0.870 | 0.920 |

Overall categorical accuracy: **86.1%** (loss 0.596). NGB vs NGS (band vs segmented neutrophils) is the main confusion.

Files: `Models/Final-Model.hdf5` (full model, used by `Model_Predict.py`) and `Weights/MV3_Weights.weights.h5` (weights only; load with `MV3_WIDTH_DIV=2 python3 Model_Test.py`). `VGG16_Weights.hdf5` from the original repo was never uploaded and is not included.

### Retraining
1. Download the BMC dataset (link above) into `./BMC-Dataset/` (one folder per class: BLA, LYT, NGB, NGS).
2. `python3 -m Utility_Functions.prepare_dataset` creates the rotated NGB images listed in `train_filenames.txt`.
3. Train: `MV3_WIDTH_DIV=2 python3 Model_Train.py` (omit the variable for the paper's full-width model; a GPU is strongly recommended). Optional env vars: `MV3_EPOCHS`, `MV3_NOISE`, `MV3_L2`, `MV3_LRDECAY=1`, `MV3_CKPT`, `MV3_BACKUP` (training resumes automatically from the backup directory).
4. Evaluate: `MV3_WIDTH_DIV=2 python3 Model_Test.py`.

`Train_Colab.ipynb` is the notebook used to train the included model on Google Colab (expects a `mv3-repo.zip` of this repo plus the prepared dataset in Drive at `MyDrive/mv3/`).

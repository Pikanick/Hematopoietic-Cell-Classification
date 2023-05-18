## ENSC 413 - Final Project Code Getting Started
This small code base cotains the training, testing, and prediction scripts for classification of white bloods cells using a convolution neural network. Before proceeding please be aware the [BMC-Dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) by Matek et al. is required for training and testing. However, for inference you can run `python3 Model_Predict.py` .
The script will load a trained model and make a prediction on the file(s) contained within `./Sample_Images/` . For fun try changing line 11 in `Model_Predict.py` to attempt at classifying different cells. Finally, a Tesla EV image was added; try seeing what cell class our CNN thinks a Tesla Model S is.

### Project Layout
- `Models/` is a directory containing our saved MV3 final model with weights. This is the same model that was tested in Section V of our paper Investigating CNN Viability and Performance within Hematopoietic Cell Classification/
- `Weights/` is a directroy containing MV3 weights and trained, on a varaint of the BMC dataset, VGG16 weights.
- `Sample_Images/` is a directory of sample images for inferance. 
- `Utility_Functions` is a directroy of different scripts for pre-processing,
    augmentation, and creation of a data generator.
- `Model_Train.py` is a script for training the final proposed MV3 architecture in "Investigating CNN Viability and Performance within Hematopoietic Cell Classification".
- `Model_Test.py` was used to obtain test metrics for final model comparison.
- `Mode_Predict.py` is a fun script to predict images contained within `Sample_Images`.

# A semi-supervised framework for digital rock image segmentation.

Repository Structure
- #1 'model' folder: Basic model implementation.
- #2 'pretrained' folder: Pretrained model weights.
- #3 'splits' folder: Produce a file specifying the ordering of the generated images.
- #4 'synthetic' folder: Synthetic images with its entropy map and mask.
- #5 'VOCdevkit' folder: The entire dataset is organized in accordance with the Pascal VOC dataset format.

Prerequisites
To run the project, you will need a conda environment that supports tensorflow2.4 and pytorch 1.10.

All codes outline 
- train.py // running the code for the entire training process. 
- configs.py // default training parameter settings.
- entropy.py // computing uncertainty map for each synthetic image.
- inference.py // generating predictions for the synthetic images to produce corresponding pseudo-labels along with their confidence scores.
- calculate-metric.py // computing segmentation metrics e.g.mIoU.



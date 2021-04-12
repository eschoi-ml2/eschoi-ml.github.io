[<- go back to the previous page](../chestxray/README.md)


# 4 key challenges
1. Multi-label classification
2. Class imbalance
3. Data leakage
4. Computation ability for big data (>100K)


## 1. Multi-label classification

## 2. Class imbalance

## 3. Data leakage
- Patient overlap between train set and valid/test set can lead to data leakage
  * The number of total images: 111,863
  * The number of unique patients: 30,773
  * The average number of images per patient: 3.64
- Split train and test set by **Patient ID**, not by Image ID


## 4. Computation ability for big data (>100K)
- Try using a TPU accelerator if TensorFlow is your deep learning framework
 > ***Tensor Processing Unit (TPU)** is an AI accelerator application-specific integrated circuit (ASIC) developed by Google specifically for neural network machine learning, particularly using Google's own TensorFlow software.*
- Training time per epoch
GPU | TPU
:-----: | :-----:
90 minutes | 90 seconds

# Bird_Classification_AI_Project_COMP_6721

## Problem Statement and Introduction
<p align="justify">
Birds are a crucial component of global biodiversity, and many species are threatened or endangered. These species
are indigenous to a certain area of the nation, hence it is important to track and estimate their populations as precisely
as possible . A significant number of rare bird species have unintentionally been killed by wind turbines in various nations.
</p>
<p align="justify">
Bird species recognition is a difficult task because of the varied illumination and multiple camera view
positions . Birds differ visually significantly across and among species. Hence, it is difficult to create models that
can precisely recognize and distinguish various species asthey have a variety of sizes, forms, colors, and other physical characteristics.
</p>

## Dataset
For this application we have selected the dataset of Indian bird specie from Kaggle. Each photo was hand-picked and taken from the eBird Platform. A total of 22.6k photos, representing all 25 distinct bird species.

The given data set consists of around 925 images of 25 different kinds of species of birds. Images are taken from the bird conservation community Platform and most of them consist of 1200x800 resolution.

**Dataset Link : https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images**

*Note : For our project we have used a set of this dataset to train and evaluate our model*

## Preprocessing the Dataset
first download the dataset from the provided link from the kaggle, and run the "data_preprocessing.py"
This python script will resize the original dataset images and it will also split the dataset into test(15%), valid(15%) and train dataset(70%). adjust the split ration as per your requirements.

Some of the image has incorrect sRBG format that can't be interpreted by the open-cv in python. so this script will remove noisy data. Deep Learning model requires a concstant input dimensionality, there we have resized the original images into 224x224 fixed resolution.

## Experiment Setup
### Hardware Setup
Training the model requires the extremely intesive processing hardware units. For this project we have used the Google Colab to get the benefits of cloud computing and highly processing unit. Here are the detailed specifications of the goolge collab hardware

| Name | Specification |
|:----:|:-------------:|
| GPU | Tesla T4 - 15.109 GB |
| CPU | Intel(R) Xeon(R) CPU @ 2.30GHz |
| CPU Frequency | 2000.178 MHz |
| RAM | 12.7 GB |
| Disk Space | 107.7 GB |

### Dataset Linking
Google collab provide the functionlity to directly read, write data from your google drive, for this project we have uploaded our training dataset on the google drive, from where we read data and save the model, and evalution metrices on the google drive. in the code just provide the dataset path of your drive and authenticate the drive permission and you are ready to train your model.


**Trained Resnet Model Link : https://drive.google.com/file/d/1vQtUnZYnJhiERQC0GV9m1MwaDedHsUFq/view?usp=drive_link**







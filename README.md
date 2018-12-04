## Honey-Bee-Images-Classfication
### Author: Xiaochi Ge, Phoebe Wu, Ruyue Zhang, Yijia Chen

The dataset is from kaggle, The BeeImage Dataset: Annotated Honey Bee Images: https://www.kaggle.com/jenny18/honey-bee-annotated-images.

It contains 5,172 bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.

#### Getting Started:
##### 1. Download the Data
Please download the dataset here:
https://console.cloud.google.com/storage/browser/group4_data

To view an image, run show_img.py
 
##### 2. EDA

Please run EDA.py

##### 3. Modelling
We use CNN to classify bee subspecies and hive health by 2 frameworks Keras and Pytorch. 
   - Keras:  
     - Subspecies_Keras.py
     - HiveHealth_Keras.py

   - Pytorch: 
     - Subspecies_Torch.py
     - HiveHealth_Torch.py
    

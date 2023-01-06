# <ins> Machine learning tools for autonomous chemistry laboratories <ins>

## Overview - Automated chemistry labs need techniques to measure material properties that are fast, non-invasive and adaptable to robotic environments. Using computer vision, we can estimate the properties of liquids by tracking fluid motion under dynamic conditions (manipulation via robot arm). 



## <ins> Description of projects: <ins>

## 1 - Autonomous data collection

Training the Convolutional Neural Network (CNN) requires a large dataset of videos consisting of moving liquids with different physical properties. We built an autonomous workflow to manipulate samples, capture the motion and store into our dataset for training later.

![alt text](./figs/auto_dc.png)
  
- [**robot_positions.py**](Autonomous_Data_Collection/robot_positions.py) - this file gives the spatial coordinates of the robot arm for picking, rotating and placing vials in the workflow
  
- [**records_vids_classes.py**](Autonomous_Data_Collection/records_vids_classes.py) - classes for recording videos of liquids for different vial sizes and video length
  
- [**run_routine.py**](Autonomous_Data_Collection/run_routine.py) - running workflow

## 2 - Train 3D-CNN
  
Once the dataset was created, we trained a 3-dimensional CNN to estimate liquid properties, either by **classification** into categories or **regression** (estimate a value). For preliminary results, see [results](./figs/results_hpo.png)
  
- [**3DCNN.py**](Train_3DCNN/3DCNN.py) - python script for training model
  
- [**functions.py**](Train_3DCNN/functions.py) - these are the 3DCNN functions required to run model
  
## 3 - Model inference
  
Once we acquired the dataset and trained our models, we developed an inference model to predict properties of new liquids
![alt text](./figs/infer.png)
  
- [**inference.py**](Inference/inference.py) - code for running inference (3DCNN classes are for different model types depending on application)
- [**functions.py**](Inference/functions.py) - these are the 3DCNN functions required to run model


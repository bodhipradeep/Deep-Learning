# Pneumonia Detection from Chest X-rays
### Overview
This project demonstrates a deep learning-based approach to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The dataset is sourced from the Kaggle Chest X-ray Pneumonia Dataset.

### Dataset
The dataset is organized into three directories:

Train: Contains X-ray images for training.   
Validation: Contains X-ray images for model validation.   
Test: Contains X-ray images for testing.   
Each directory has two categories:

NORMAL: Images of healthy lungs.  
PNEUMONIA: Images showing signs of pneumonia.    

EDA and Visualization
Sample Images
Below are some sample images from the Normal and Pneumonia categories in the training set:
Normal
![image](https://github.com/user-attachments/assets/e28e5d84-3c8d-4ded-94fd-092bb46b6cea)

Pneumonia
![image](https://github.com/user-attachments/assets/60b42953-406e-40be-869a-119580a82e6d)

Image Distribution in Dataset
![image](https://github.com/user-attachments/assets/f60f68e9-8ccb-44c0-9630-4827b8176907)

Pie Chart of Image Distribution
![image](https://github.com/user-attachments/assets/0fb85d52-3a36-41c4-8725-2676697672ea)

Data Augmentation and Preprocessing
We used ImageDataGenerator to apply data augmentation techniques like rotation, zoom, shear, and horizontal flipping. This helps improve the generalization of the model.

Model Architecture   
The CNN architecture includes:   

Convolutional Layers with ReLU activation   
MaxPooling for down-sampling   
Batch Normalization for faster convergence   
Dropout for regularization   
Fully Connected Layers with Sigmoid activation for binary classification   
Training was conducted with EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model based on validation loss.  


Evaluation
Test Accuracy: 86.86%

Confusion Matrix
![image](https://github.com/user-attachments/assets/8aa77254-c7bc-40ea-b021-402d2e23751b)

Plot Training and Validation Accuracy & Loss
![image](https://github.com/user-attachments/assets/5f290731-7420-48d8-91a6-2cdf67fcaa1d)

Predictions on Test Set   
Below are some sample predictions:
![image](https://github.com/user-attachments/assets/ce9bc007-0973-4409-87f7-e04a70934361)

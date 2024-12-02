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

Normal	Pneumonia
Image Distribution in Dataset

Pie Chart of Image Distribution

Data Augmentation and Preprocessing
We used ImageDataGenerator to apply data augmentation techniques like rotation, zoom, shear, and horizontal flipping. This helps improve the generalization of the model.

Model Architecture
The CNN architecture includes:

Convolutional Layers with ReLU activation
MaxPooling for down-sampling
Batch Normalization for faster convergence
Dropout for regularization
Fully Connected Layers with Sigmoid activation for binary classification
Model Summary:

markdown
Copy code
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
 ...
 dense_1 (Dense)             (None, 1)                 129       
=================================================================
Total params: 427,329
Trainable params: 426,561
Non-trainable params: 768
Training and Validation
Training was conducted with EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model based on validation loss.


Evaluation
Test Accuracy: xx%

Confusion Matrix

Classification Report
markdown
Copy code
              precision    recall  f1-score   support

      Normal       0.xx      0.xx      0.xx       xxx
   Pneumonia       0.xx      0.xx      0.xx       xxx

    accuracy                           0.xx       xxx
   macro avg       0.xx      0.xx      0.xx       xxx
weighted avg       0.xx      0.xx      0.xx       xxx
Predictions on Test Set
Below are some sample predictions:

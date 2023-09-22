# BMI-prediction-using-VGG-Face
Real-Time BMI Estimation using Computer Vision and Deep Learning: A Web API Approach for Facial Image Analysis


Obesity is a prevalent public health issue with significant implications for chronic diseases such as cardiovascular disorders, diabetes, and certain types of cancer. Monitoring Body Mass Index (BMI) is a valuable indicator of overall health and fitness. However, conventional BMI measurement methods often prove incon- venient or require specialized equipment, hindering widespread adoption and regular monitoring. To address this challenge, we aim to develop a web API that leverages computer vision techniques to predict BMI in real-time using facial images. By harnessing the power of computer vision and deep learning, our solution aims to provide an efficient and practical tool for BMI estimation, enabling personalized health monitoring for developers and end-users alike. The problem we aim to solve is the inconvenience and lack of accessibility associ- ated with conventional BMI measurement methods. Current approaches, such as using scales or specialized equipment, can be cumbersome, time-consuming, and may not be readily available to everyone. Additionally, these methods of- ten require individuals to disrobe or provide personal information, which may deter some users from regular monitoring. Our goal is to overcome these limi- tations by developing a web API that allows users to estimate their BMI using
facial images. By providing the option to upload static facial images or cap- ture real-time images through a webcam, we eliminate the need for specialized equipment and simplify the process for users. The web API empowers devel- opers and end-users to integrate BMI prediction functionality and personalized health monitoring seamlessly.



## Data Wrangling and Analysis for BMI Prediction

### Introduction
This project provides an overview of the data preprocessing and analysis for predicting Body Mass Index (BMI) based on facial images. We use a dataset containing images and their corresponding BMI values. Here's a breakdown of the steps carried out in the provided code:

### Libraries

Various libraries are imported for processing and modeling:

1. pandas, numpy: Data manipulation.
2. tensorflow: Deep learning framework.
3. os, glob: File handling.
4. cv2: Image processing.
5. seaborn, matplotlib: Visualization.
6. sklearn: Machine learning utility functions.
7. keras_vggface: Utility for face-based applications.
8. PIL: Image manipulation.
9. scipy: Statistical computations.

### Data Loading and Initial Exploration
1. Loading Data: The dataset is loaded from the data.csv file.
2. Inspecting Data: The first few rows of the dataset are displayed to inspect its structure.
3. Train-Test Split: The dataset is split into training and test sets based on the is_training column.
4. Image Existence Check: Both training and test data are filtered to ensure all referenced images exist in the directory.
5. Dataset Shape: The shape of the training and test datasets is printed to check the number of samples.
6. Null Values Check: The number of null values in the training data is checked.
7. BMI Distribution: The distribution of BMI values in both training and test sets is visualized using histograms.


### Image Processing

1. Loading and Resizing Images:
  1. For the training and test datasets, images referenced by the name column are loaded.
  2. Images are resized to (224, 224) to be compatible with pre-trained models.
     
2. Normalization: Image pixel values are normalized to the range [0, 1] by dividing by 255. This is a common preprocessing step to make the neural network converge faster.
3. Extracting BMI Values: Corresponding BMI values are extracted for each image from the dataset.
4. Final Shape: The shape of the processed image data and BMI values for training and test sets are printed for verification.


## BMI Prediction using VGG16 Model

This project uses a pre-trained VGG16 model from the VGGFace library to predict Body Mass Index (BMI) based on images.

### Project Overview:

1. VGG16 Model Customization: The base VGG16 model is taken from VGGFace with weights pre-trained on faces. We remove the top classification layers and replace them with our custom fully connected layers to regress on BMI values.

2. Training and Evaluation: The custom VGG16 model is trained using the Mean Squared Error (MSE) loss given that this is a regression task. We also employ early stopping to halt the training when the model's performance stops improving on the validation data.

3. Performance Metrics: The performance of the model is evaluated using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

4. Visualization: The predicted BMIs are plotted against the actual BMIs for visual assessment of the model's performance.

### Code Structure:

1. Model Definition:
  1. Load the VGG16 model without the top layers.
  2. Freeze all layers except the last three.
  3. Add custom dense and dropout layers.
  4. Compile the model using Adam optimizer and MSE loss.

2.Training:
  1. Use an early stopping callback to monitor the MSE and stop training when the model stops improving.
  2. Train the model on training data.

3. Evaluation:
  1. Evaluate the model's performance on both training and test data.
  2. Compute and display performance metrics including MAE, MSE, RMSE, and R2.

4. Visualization:
  1. Plot actual vs. predicted BMI values for the test data.

5. Model Persistence:
  1. Save the trained model to a .h5 file for future use.

### Requirements:
1. VGGFace
2. TensorFlow
3. Keras
4. NumPy
5. Matplotlib
6. Scikit-learn

### How to Run:

1. Ensure you have all the required libraries installed.
2. Load your training and test data (not provided in the code snippet).
3. Run the given code to train and evaluate the model.
4. Visualize the model's predictions against actual values.
5. Save the trained model for future use.
   
### Results:

The model's performance can be assessed visually by the plotted graph showing actual versus predicted BMIs and quantitatively using the printed evaluation metrics.



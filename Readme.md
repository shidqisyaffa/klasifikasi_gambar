Traffic Sign Classification Project
Overview
This project implements a convolutional neural network (CNN) for classifying traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model can identify 43 different classes of traffic signs with high accuracy.
Features

Deep learning model for traffic sign classification
Dataset split into train, validation, and test sets
Comprehensive model evaluation
Model export in multiple formats (H5, SavedModel, TF-Lite, TensorFlow.js)
Performance visualization and metrics

Dataset
The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset which contains over 50,000 images of 43 different classes of traffic signs.
Data Preprocessing

Images are resized to 30x30 pixels
Data is split into:

Training set (70%)
Validation set (15%)
Test set (15%)



Model Architecture
The model consists of:

4 Convolutional layers with increasing filters (32, 64, 128, 256)
2 MaxPooling layers
Dropout layers (to prevent overfitting)
Dense layers for classification

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        2432      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 22, 22, 64)        51264     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 11, 11, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 11, 11, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 9, 128)         73856     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 256)         295168    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 256)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 3, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               1180160   
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 43)                22059     
=================================================================
Total params: 1,624,939
Trainable params: 1,624,939
Non-trainable params: 0
_________________________________________________________________
Training

Optimizer: Adam
Loss function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 35
Batch size: 128
Training uses GPU acceleration when available

Performance
The model achieves:

High accuracy on both validation and test sets
Detailed classification reports and metrics
Performance visualization for training and validation

Model Exports
The trained model is exported in multiple formats:

Keras H5 format (traffic_classifier.h5)
TensorFlow SavedModel format (traffic_classifier_savedmodel)
TensorFlow Lite format (traffic_classifier.tflite)
TensorFlow.js format (tfjs_model)

Requirements
The project requires the following Python libraries:

TensorFlow
NumPy
Pandas
Pillow (PIL)
OpenCV
Scikit-learn
Matplotlib
Seaborn

A complete list of requirements is available in the requirements.txt file.
Usage

Ensure all dependencies are installed:
pip install -r requirements.txt

Run the training script:
python train.py

The script will:

Load and preprocess the dataset
Train the model
Evaluate performance
Save the model in multiple formats


The saved models can be used for inference:
python# Example using the H5 model
from tensorflow.keras.models import load_model
model = load_model('saved_models/traffic_classifier.h5')

# Make predictions
predictions = model.predict(images)


Potential Applications

Self-driving cars
Driver assistance systems
Traffic monitoring
Road safety applications
Educational tools for learning traffic signs

Future Improvements

Data augmentation to improve robustness
Transfer learning with pre-trained models
Hyperparameter tuning
Additional post-processing techniques
Converting to optimized models for embedded systems

License
This project is available under the MIT License.

## 👨‍💻 Author

Shidqi Ahmad Musyaffa'

---


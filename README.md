This project implements a Convolutional Neural Network (CNN) using PyTorch to detect human emotions from facial images.
The model classifies facial expressions into 7 emotion categories.

Dataset

The model uses the FER2013 dataset.

Emotion classes:
	•	Angry
	•	Disgust
	•	Fear
	•	Happy
	•	Sad
	•	Surprise
	•	Neutral

Dataset structure:
data/
 ├── train/
 │   ├── angry
 │   ├── disgust
 │   ├── fear
 │   ├── happy
 │   ├── sad
 │   ├── surprise
 │   └── neutral
 └── test/

 Technologies Used
	•	Python
	•	PyTorch
	•	Torchvision
	•	NumPy
	•	OpenCV

Model Architecture

The CNN model consists of:
	•	Convolution Layers
	•	ReLU Activation
	•	Max Pooling
	•	Fully Connected Layers
	•	Softmax Output

Pipeline:
Face Image → CNN → Emotion Prediction

How to Run

Install dependencies:
pip install torch torchvision numpy matplotlib pandas opencv-python

Train the model:
python train.py


Complete architecture

Results 
Expected accuracy:

60% – 70% (basic CNN on FER2013)

Author

Gagan K
CSE (AI & ML)
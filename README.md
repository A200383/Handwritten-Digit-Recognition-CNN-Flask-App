# Handwritten Digit Recognition with CNN and Flask

## Introduction
This project demonstrates the development and deployment of a deep learning model for handwritten digit recognition. The system leverages a **Convolutional Neural Network (CNN)** built with the **PyTorch** library to classify handwritten digits from the **MNIST dataset**.  
The trained model is then deployed as a web application using the **Flask** framework, allowing users to upload an image of a handwritten digit and receive a real-time prediction.

---

## Dataset
The project utilizes the **MNIST (Modified National Institute of Standards and Technology)** dataset — a classic dataset in computer vision and machine learning.

- **Content:** 60,000 training images and 10,000 testing images. Each image is a **28×28 pixel grayscale** image of a handwritten digit.
- **Classes:** 10 classes (digits **0 through 9**).
- **Source:** [Download MNIST Dataset](https://drive.google.com/drive/folders/1z4iFh1gJiRS3BpdzhYwf9tZGbh__CDNg?usp=sharing)
- **Setup:** Download the dataset and place it in a folder named **`data`** in the project's root directory.

---

## Model Architecture
A **Convolutional Neural Network (CNN)** is employed for its high accuracy in image classification tasks.

Key components:
- **Convolutional Layers:** Learn features such as edges, textures, and patterns.
- **Pooling Layers:** Reduce feature map size and increase robustness.
- **Activation Functions:** Use non-linear functions like **ReLU** to learn complex patterns.
- **Fully Connected Layers:** Perform final classification based on extracted features.
- **Output Layer:** Softmax activation for probability distribution over 10 classes.

The model is implemented in **PyTorch**.
---

## Getting Started

### 1. Clone the repository
```bash
git clone buhttps://github.com/A200383/Handwritten-Digit-Recognition-CNN-Flask-App
cd Handwritten-Digit-Recognition-CNN-Flask-App

2. Download the dataset

Download MNIST.zip from the provided Google Drive link.

Create a folder named data in the root directory.

Extract the dataset into this folder.


3. Install dependencies

It is recommended to use a virtual environment.

pip install -r requirements.txt

(Dependencies include: torch, torchvision, flask, numpy, pillow, etc.)

4. Train the model

python model.py

This will generate a model file (e.g., mnist_model.pth) containing trained weights.

5. Run the Flask application

python app.py

6. Access the web app

Open your browser and visit:

http://127.0.0.1:5000


---

Usage

Upload an image of a handwritten digit.

The app displays the uploaded image and predicts the digit.

Shows prediction confidence score.



---

Conclusion

This project showcases an end-to-end machine learning pipeline, from data preprocessing and model training to deployment. It highlights:

The effectiveness of CNNs in image classification.

Practical integration of deep learning models into Flask web apps.





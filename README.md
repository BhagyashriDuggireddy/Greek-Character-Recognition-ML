# Performance-Evaluation-and-Comparative-Analysis-of-Machine-Learning-Models-for-Handwritten-Greek-Character-Recognition

## Introduction

This research paper focuses on the domain of Performance Evaluation and Comparative Analysis of Machine Learning Models for Handwritten Greek Character Recognition, specifically addressing the identification and classification of diverse Greek characters. The primary objective is to conduct a comprehensive performance evaluation and comparison of three distinct machine learning algorithms: Convolutional Neural Network (CNN), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) classifier.

## Dataset

To initiate the research, we carefully curated a dataset consisting of grayscale images representing 24 unique Greek characters. These characters exhibited a wide range of complexity and stylistic variations, posing significant challenges for our chosen algorithms. The dataset was thoughtfully split into separate training and testing sets, and all images were uniformly resized to a 14x14 pixel dimension. To ensure data consistency, pixel values were normalized, and character labels were transformed into a suitable format for categorical classification.

## Machine Learning Models

Our exploration of machine learning models commenced with the construction of a Convolutional Neural Network (CNN) using the Keras library. This deep neural network was meticulously designed with convolutional layers to extract intricate features from the input images. Max-pooling layers further distilled this information, resulting in a robust representation for each character. The CNN model underwent an extensive training regimen spanning 35 epochs, with early stopping mechanisms implemented to mitigate overfitting.

Subsequently, we delved into the realm of Support Vector Machine (SVM), a well-established machine learning approach. Using the Scikit-learn framework, we standardized the image data before feeding it into the SVM model equipped with a radial basis function (RBF) kernel. This configuration facilitated character discrimination by mapping them into higher-dimensional feature spaces. Similarly, the K-Nearest Neighbors (KNN) classifier adopted a proximity-based classification approach, making predictions based on the closest neighbors of each data point.

## Performance Evaluation

Integral to our methodology was the rigorous evaluation of the algorithms' performance. Each model underwent comprehensive assessments on both validation and test datasets. Notably, the CNN,SVM and KNN models provided confidence scores, offering insights into the models' certainty in their predictions and enhancing transparency in their decision-making processes.

## Results

The culmination of our research was a meticulous comparison of the test accuracies achieved by each algorithm. To provide an intuitive understanding of their relative strengths, we presented a succinct bar graph, offering a visual summary of their performance levels.




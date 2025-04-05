# CIFAR-100 Image Classification with MobileNetV2 and Transfer Learning

This project demonstrates image classification on the CIFAR-100 dataset using a pre-trained MobileNetV2 model and transfer learning techniques. It involves loading the dataset, performing data augmentation, fine-tuning the model, and evaluating its performance.

## Code Structure and Logic

1. **Data Loading and Preprocessing:**
   - The CIFAR-100 dataset is loaded using `cifar100.load_data()`.
   - The labels are one-hot encoded using `to_categorical()`.
   - Images are resized and normalized using a `preprocess` function.
   - `tf.data` is used to create efficient input pipelines with batching and prefetching.

2. **Transfer Learning with MobileNetV2:**
   - A pre-trained MobileNetV2 model is loaded with weights from ImageNet.
   - The top classification layer is removed.
   - The base model is initially frozen (`trainable = False`) to preserve pre-trained weights.
   - A new classification head is added, consisting of a global average pooling layer, a dense layer with 512 units and ReLU activation, a dropout layer for regularization, and a final dense layer with 100 units and softmax activation.

3. **Model Training and Fine-tuning:**
   - The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
   - It is trained for 10 epochs using `model.fit()`.
   - After initial training, the base model is unfrozen (`trainable = True`) for fine-tuning.
   - The first 50 layers of the base model are frozen to prevent overfitting.
   - The model is recompiled with a lower learning rate and trained for 5 more epochs.

4. **Evaluation and Visualization:**
   - The model's performance is evaluated on the test set using `model.evaluate()`.
   - Learning curves (accuracy and loss) are plotted to visualize the training process.
   - A confusion matrix is generated to analyze the model's predictions.
   - Images from a specific class are displayed along with their predicted labels.
   
![image](https://github.com/user-attachments/assets/998b8750-d6cf-4635-b5e1-ce04c7252a2a)

## Technology and Algorithms

- **TensorFlow and Keras:** Used for building and training the neural network model.
- **MobileNetV2:** A pre-trained convolutional neural network architecture used as the base model.
- **Transfer Learning:** Leveraging pre-trained weights to improve performance on a new task.
- **Data Augmentation:** Resizing and normalizing images to increase the dataset size and improve model robustness.
- **Adam Optimizer:** An optimization algorithm for training the model.
- **Categorical Cross-Entropy Loss:** A loss function suitable for multi-class classification.
- **Confusion Matrix:** A visualization tool for evaluating the performance of a classification model.
- **Matplotlib and Seaborn:** Used for creating plots and visualizations.

  ![image](https://github.com/user-attachments/assets/58876602-5cf7-441f-ae27-b5330ca9c457)

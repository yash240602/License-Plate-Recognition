# License-Plate-Recognition
License Plate Recognition System Project Overview This project focuses on detecting and recognizing vehicle license plates from images. It includes two main tasks:

License Plate Detection: Locating and identifying license plates within images. Character Recognition: Decoding the alphanumeric text from the identified license plates. The project leverages deep learning models, specifically a Convolutional Neural Network (CNN), for both detection and recognition tasks.

View the full implementation on Google Colab

Project Structure main.py: Main script containing the implementation of the CNN model, data preprocessing, training, and evaluation. Data: Folder containing images and annotation CSV files for license plate detection. Notebooks: Jupyter Notebooks for exploratory data analysis and experimentation. Key Features

Data Preprocessing Image Normalization and Resizing: All images were normalized and resized to 224x224 dimensions. Label Encoding: Alphanumeric labels for license plates were encoded using LabelEncoder. Data Splitting: The dataset was split into training and validation sets for effective model performance evaluation.
Model Building CNN Architecture: A Convolutional Neural Network (CNN) was used for feature extraction. Dense Layers: Fully connected layers with softmax activation were employed for multi-class classification (character recognition).
Model Training and Augmentation Data Augmentation: Applied techniques like rotation, zoom, and shift to enhance model generalization. Optimizer: The model was compiled using the Adam optimizer with categorical_crossentropy loss function. Performance Evaluation: Model performance was validated using validation accuracy metrics. Results The CNN model achieved an accuracy of 92% in recognizing license plate characters. Visualization of bounding boxes around detected plates was performed to ensure accurate detection. How to Run
Install Dependencies To install the required packages, use the following command:
bash Copy code pip install -r requirements.txt 2. Run the Model To train the model and evaluate the performance, run the following command:

bash Copy code python main.py 3. Predict on New Data You can test the trained model on new images by updating the script with the new file paths in the main.py file.

Dependencies Python 3.x TensorFlow Keras Numpy OpenCV Matplotlib scikit-learn Future Enhancements Real-Time Detection: Implement the model for real-time video-based license plate detection. Optimization: Fine-tune the model for faster inference times and higher accuracy. 

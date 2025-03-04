import os
import random
from matplotlib import pyplot as plt
import cv2

import numpy as np
import pandas as pd

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.metrics import Precision, Recall

import keras_tuner as kt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import os
import zipfile

# Define paths
zip_path = '/content/dataset.zip'
top_dir = '/content/dataset'

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(top_dir)

# Define train and test directories after extraction
train_dir = os.path.join(top_dir, 'train')
test_dir = os.path.join(top_dir, 'test')

# List REAL and FAKE images in the training directory
train_real = os.listdir(os.path.join(train_dir, 'REAL'))
train_fake = os.listdir(os.path.join(train_dir, 'FAKE'))


# List REAL and FAKE images in the testing directory
test_real = os.listdir(os.path.join(test_dir, 'REAL'))
test_fake = os.listdir(os.path.join(test_dir, 'FAKE'))

print("Number of real images in training set:", len(train_real))
print("Number of fake images in training set:", len(train_fake))
print("Number of real images in testing set:", len(test_real))
print("Number of fake images in testing set:", len(test_fake))











import tensorflow as tf  # Import TensorFlow
from keras.utils import image_dataset_from_directory

# Load training and validation datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    label_mode='binary',
    batch_size=32,
    image_size=(32, 32)
)

val_dataset = image_dataset_from_directory(
    test_dir,
    label_mode='binary',
    batch_size=32,
    image_size=(32, 32)
)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

print("Training and validation datasets loaded successfully.")



def create_cnn_model(filters, layers):
    """
    Creates a CNN model with the specified number of filters and layers.
    Args:
        filters (int): Number of filters in the convolutional layers.
        layers (int): Number of convolutional layers.

    Returns:
        Sequential: Compiled CNN model.
    """
    model = Sequential()

    # Rescaling layer
    model.add(Rescaling(1./255, input_shape=(32, 32, 3)))

    # Add convolutional and pooling layers
    for _ in range(layers):
        model.add(Conv2D(filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

    # Flatten and output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    return model





# Define topologies: [(filters, layers)]
topologies = [
    (16, 1), (16, 2), (16, 3),
    (32, 1), (32, 2), (32, 3),
    (64, 1), (64, 2), (64, 3),
    (128, 1), (128, 2), (128, 3)
]

results = []

# Train and evaluate each topology
for filters, layers in topologies:
    print(f"Training model with {filters} filters and {layers} layers...")

    # Create model
    model = create_cnn_model(filters, layers)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        verbose=2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    # Evaluate model
    val_metrics = model.evaluate(val_dataset, verbose=0)
    precision = val_metrics[2]
    recall = val_metrics[3]
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Append results
    results.append({
        'Filters': filters,
        'Layers': layers,
        'Accuracy': val_metrics[1],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Validation Loss': val_metrics[0]
    })

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results)






# Print table for Accuracy
print("Validation Accuracy for Different Topologies:")
accuracy_table = results_df.pivot_table(index='Filters', columns='Layers', values='Accuracy', aggfunc='max')
print(accuracy_table)

# Print table for Validation Loss
print("\nValidation Loss for Different Topologies:")
loss_table = results_df.pivot_table(index='Filters', columns='Layers', values='Validation Loss', aggfunc='min')
print(loss_table)

# Print table for Precision
print("\nValidation Precision for Different Topologies:")
precision_table = results_df.pivot_table(index='Filters', columns='Layers', values='Precision', aggfunc='max')
print(precision_table)

# Print table for Recall
print("\nValidation Recall for Different Topologies:")
recall_table = results_df.pivot_table(index='Filters', columns='Layers', values='Recall', aggfunc='max')
print(recall_table)

# Print table for F1 Score
print("\nValidation F1 Score for Different Topologies:")
f1_table = results_df.pivot_table(index='Filters', columns='Layers', values='F1 Score', aggfunc='max')
print(f1_table)



# Sort by Accuracy (descending) and Validation Loss (ascending)
best_topology_df = results_df.sort_values(by=['Accuracy', 'Validation Loss'], ascending=[False, True])

# Extract the best topology
best_topology = best_topology_df.iloc[0]

# Print the details of the best-performing topology
print("Topology with Highest Accuracy and Minimal Loss:")
print(f"Filters: {best_topology['Filters']}")
print(f"Layers: {best_topology['Layers']}")
print(f"Accuracy: {best_topology['Accuracy']:.4f}")
print(f"Precision: {best_topology['Precision']:.4f}")
print(f"Recall: {best_topology['Recall']:.4f}")
print(f"F1 Score: {best_topology['F1 Score']:.4f}")
print(f"Validation Loss: {best_topology['Validation Loss']:.4f}")



import tensorflow as tf
from tensorflow.keras import layers, models

# Define the feature extractor with 128 filters and 3 layers
def build_feature_extractor():
    feature_extractor = models.Sequential([
        layers.Rescaling(1./255, input_shape=(32, 32, 3)),  # Normalize input
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])
    return feature_extractor



# Function to build the dense model with varying architectures
def build_dense_model(feature_extractor, neurons, num_layers, activation):
    model = models.Sequential()
    model.add(feature_extractor)  # Add the feature extractor
    for _ in range(num_layers):
        model.add(layers.Dense(neurons, activation=activation))  # Add dense layers with given neurons and activation
    model.add(layers.Dense(1, activation='sigmoid'))  # Final output layer with sigmoid activation for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define the list of neurons and dense layers to explore
neurons_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
dense_layers = [1, 2, 3]

# Store results
results = []

# Load training and validation datasets (Ensure X_train, y_train, X_test, y_test are ready)
# Example: X_train, y_train, X_test, y_test = your_dataset_loading_function()



# Extract features and labels from the train and validation datasets
def extract_data_from_dataset(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

# Extract data
X_train, y_train = extract_data_from_dataset(train_dataset)
X_test, y_test = extract_data_from_dataset(val_dataset)






from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Iterate through different topologies and evaluate
for neurons in neurons_list:
    for num_layers in dense_layers:
        # Train with ReLU activation
        model_relu = build_dense_model(build_feature_extractor(), neurons, num_layers, 'relu')
        model_relu.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)
        y_pred_relu = model_relu.predict(X_test).round()  # Get binary predictions

        # Calculate metrics for ReLU model
        accuracy_relu = accuracy_score(y_test, y_pred_relu)
        precision_relu = precision_score(y_test, y_pred_relu)
        recall_relu = recall_score(y_test, y_pred_relu)
        f1_relu = f1_score(y_test, y_pred_relu)
        loss_relu = model_relu.evaluate(X_test, y_test, verbose=0)[0]

        results.append({
            'Neurons': neurons,
            'Dense Layers': num_layers,
            'Activation': 'ReLU',
            'Accuracy': accuracy_relu,
            'Loss': loss_relu,
            'Precision': precision_relu,
            'Recall': recall_relu,
            'F1 Score': f1_relu
        })

        # Train with Sigmoid activation
        model_sigmoid = build_dense_model(build_feature_extractor(), neurons, num_layers, 'sigmoid')
        model_sigmoid.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)
        y_pred_sigmoid = model_sigmoid.predict(X_test).round()

        # Calculate metrics for Sigmoid model
        accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
        precision_sigmoid = precision_score(y_test, y_pred_sigmoid)
        recall_sigmoid = recall_score(y_test, y_pred_sigmoid)
        f1_sigmoid = f1_score(y_test, y_pred_sigmoid)
        loss_sigmoid = model_sigmoid.evaluate(X_test, y_test, verbose=0)[0]

        results.append({
            'Neurons': neurons,
            'Dense Layers': num_layers,
            'Activation': 'Sigmoid',
            'Accuracy': accuracy_sigmoid,
            'Loss': loss_sigmoid,
            'Precision': precision_sigmoid,
            'Recall': recall_sigmoid,
            'F1 Score': f1_sigmoid
        })



import pandas as pd

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results)

# Create separate tables for each validation metric
accuracy_df = results_df[['Neurons', 'Dense Layers', 'Activation', 'Accuracy']].pivot(index='Neurons', columns='Dense Layers', values='Accuracy')
loss_df = results_df[['Neurons', 'Dense Layers', 'Activation', 'Loss']].pivot(index='Neurons', columns='Dense Layers', values='Loss')
precision_df = results_df[['Neurons', 'Dense Layers', 'Activation', 'Precision']].pivot(index='Neurons', columns='Dense Layers', values='Precision')
recall_df = results_df[['Neurons', 'Dense Layers', 'Activation', 'Recall']].pivot(index='Neurons', columns='Dense Layers', values='Recall')
f1_df = results_df[['Neurons', 'Dense Layers', 'Activation', 'F1 Score']].pivot(index='Neurons', columns='Dense Layers', values='F1 Score')

# Output all tables
print("Accuracy Metrics:")
print(accuracy_df)
print("\nLoss Metrics:")
print(loss_df)
print("\nPrecision Metrics:")
print(precision_df)
print("\nRecall Metrics:")
print(recall_df)
print("\nF1 Score Metrics:")
print(f1_df)



# Reshape the data for easy filtering and merge them
accuracy_data_melted = accuracy_df.melt(id_vars='Neurons', value_vars=['Layer 1', 'Layer 2', 'Layer 3'],
                                         var_name='Layer', value_name='Accuracy')
loss_data_melted = loss_df.melt(id_vars='Neurons', value_vars=['Layer 1', 'Layer 2', 'Layer 3'],
                                var_name='Layer', value_name='Loss')

# Combine accuracy and loss data
accuracy_data_melted['Layer'] = accuracy_data_melted['Layer'].str.extract('(\d)').astype(int)
loss_data_melted['Layer'] = loss_data_melted['Layer'].str.extract('(\d)').astype(int)
combined_data = pd.merge(accuracy_data_melted, loss_data_melted, on=['Neurons', 'Layer'])

# Sort by Loss first, and then by Accuracy
sorted_combined_data = combined_data.sort_values(by=['Loss', 'Accuracy'], ascending=[True, False])

# Get the best topology with least loss and highest accuracy
best_topology = sorted_combined_data.iloc[0]

# Print the result
print("Best Topology based on Least Loss and Highest Accuracy:")
print(f"Neurons: {best_topology['Neurons']}")
print(f"Layer: {best_topology['Layer']}")
print(f"Accuracy: {best_topology['Accuracy']:.2f}")
print(f"Loss: {best_topology['Loss']:.3f}")




import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'Images/'  # Changed to uppercase to match folder name
IMG_SIZE = 180
BATCH_SIZE = 32
EPOCHS = 15

def load_and_analyze_data():
    """Load and analyze image data from directories"""
    # Verify folder exists
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Directory {BASE_DIR} not found. Please ensure the folder exists.")
    
    # Count images
    count = 0
    dirs = os.listdir(BASE_DIR)
    for dir_name in dirs:
        files = list(os.listdir(os.path.join(BASE_DIR, dir_name)))
        print(f"{dir_name} Folder has {len(files)} Images")
        count += len(files)
    print(f"Images Folder has {count} Images")

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        seed=123,
        validation_split=0.2,
        subset='training',
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        seed=123,
        validation_split=0.2,
        subset='validation',
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE))
    
    return train_ds, val_ds

def plot_sample_images(dataset, class_names):
    """Plot sample images from dataset"""
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')
    plt.show()

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return Sequential([
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

def build_model(data_augmentation):
    """Build and compile the CNN model"""
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5)  # 5 classes for flowers
    ])
    
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds):
    """Train the model and return training history"""
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

def classify_image(model, image_path, class_names):
    """Classify a single image"""
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    result = tf.nn.softmax(predictions[0])
    confidence = np.max(result) * 100
    predicted_class = class_names[np.argmax(result)]
    
    return f"The image belongs to {predicted_class} with {confidence:.2f}% confidence"

def main():
    # Load and analyze data
    train_ds, val_ds = load_and_analyze_data()
    class_names = train_ds.class_names
    print("Class names:", class_names)
    
    # Plot sample images
    plot_sample_images(train_ds, class_names)
    
    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Build and train model
    model = build_model(data_augmentation)
    model.summary()
    
    history = train_model(model, train_ds, val_ds)
    
    # Save model
    model.save('Flower_Recog_Model.h5')
    print("Model saved successfully.")
    
    # Example classification
    sample_path = 'Sample/rose.jpg'
    if os.path.exists(sample_path):
        result = classify_image(model, sample_path, class_names)
        print(result)
    else:
        print(f"Sample image not found at {sample_path}")

if __name__ == "__main__":
    main()

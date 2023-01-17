"""
Script creado para almacenar funciones que se usan en varios notebooks
Las funciones son:
- plot_training_curves: para graficar las curvas de entrenamiento
- compare_historys: graficar curvas de entrenamiento al aplicar Fine-Tuning
- view_random_image: para visualizar una imagen aleatoria
- load_and_prep_image: para cargar y preparar una imagen para hacer una predicción
- pred_and_plot: para hacer una predicción y graficar la imagen
- create_tensorboard_callback: para crear un callback para TensorBoard
- walk_through_dir: para visualizar las imágenes de un directorio
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import tensorflow as tf
import numpy as np
import datetime



""""
Función para graficar las curvas de entrenamiento
"""
def plot_training_curves(history):
    """
    Plot training curves for accuracy and loss metrics.
    Args: 
        history object from model.fit()
    Returns: 
        Plot of training/validation loss and accuracy curves


    """
    # Plot training & validation accuracy values

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Training Curves')
    plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.grid(True)
    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    #plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.grid(True)
    plt.show()

""""
Función para graficar las curvas de entrenamiento al aplicar Fine-Tuning
"""
def compare_historys(original_history, new_history, initial_epochs = 5):
    """
    Compare two TensorFlow History objects.
    Inputs:
        original_history: history object from model.fit() with Feature Extraction
        new_history: new history object from model.fit() with Fine-Tuning
        initial_epochs = default 5
    Outputs:
        Plot of training/validation loss and accuracy curves
    """
    # Get original history measurements
    loss = original_history.history['loss']
    val_loss = original_history.history['val_loss']

    acc = original_history.history['accuracy']
    val_acc = original_history.history['val_accuracy']

    # Combine original history
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']

    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    # Plot training & validation Accuracy values
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label = "Train")
    plt.plot(total_val_acc, label = "Val")
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.title('Model Training Curves')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label = "Train")
    plt.plot(total_val_loss, label = "Val")
    
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

"""
Función para visualizar una imagen aleatoria
"""
def view_random_image(target_dir, target_class):
    """
    View a random image from a target directory.
    input: target_dir = target directory (e.g. train_dir or test_dir)
           target_class = target class (e.g. "PNEUMONIA" or "NORMAL")
    output: displays a random image and prints the image shape
    """
    # Set the target directory (we'll view images from here)
    target_folder = target_dir + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");

    print(f"Image shape: {img.shape}") # show the shape of the image

    return img

"""
Función para cargar y preparar una imagen para hacer una predicción
"""
def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels).
    input: filename = path to the image
            img_shape = desired size of the image (default = 224)
    output: preprocessed image tensor
    """
    # Leer la imagen
    img = tf.io.read_file(filename)
    # Decodificarla en un tensor
    img = tf.image.decode_image(img)
    # Cambiar el tamaño de la imagen
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Escalar la imagen (valores entre 0 y 1)
    img = img/255.
    return img    

"""
Función para hacer una predicción y graficar la imagen
"""
def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction with a model and plots the image with the predicted class as the title.
    input: model = a trained machine learning model
            filename = path to the image
            class_names = a list of class names (e.g. ['cat', 'dog'])
    output: displays an image with the predicted class as the title
    """
    # Preprocesar la imagen
    img = load_and_prep_image(filename)
    # Hacer una predicción
    pred = model.predict(tf.expand_dims(img, axis=0))
    # Obtener la clase predicha
    if len(class_names)> 2:
        pred_class = class_names[np.argmax(pred)] # Categorical prediction
    else:
        pred_class = class_names[int(tf.round(pred))] # Binary prediction
    # Cargar la imagen
    img = mpimg.imread(filename)
    # Mostrar la imagen y agregar el título
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);

"""
Función para crear un callback para TensorBoard
"""
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"

    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

"""
Función para recorrer un directorio de clasificación de imágenes y averiguar cuántos archivos (imágenes)
hay en cada subdirectorio.
"""
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
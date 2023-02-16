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
- Función para descomprimir un dataset
- Función para calcular el resultado del desempeño de un modelo
- Función para graficar la matrix de confusión
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import zipfile
import itertools

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
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
      # Rescale the image (get all values between 0 and 1)
      return img/255.
    else:
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

"""
Función para descomprimir un dataset
"""
def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()


"""
Función para calcular el resultado del desempeño de un modelo
"""
def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

"""
Función para graficar la matrix de confusión
"""
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
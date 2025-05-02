from matplotlib import pyplot as plt
import numpy as np


def evaluate_acc_loss(history, epochs):   
    # Evaluate the model
    accuracy = history.history['categorical_accuracy']
    val_accuracy = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    f1_score = np.array(history.history['f1_score']).transpose()
    val_f1_score = np.array(history.history['val_f1_score']).transpose()

    epochs_count = range(1, epochs + 1)   
    
    return accuracy, val_accuracy, loss, val_loss, f1_score, val_f1_score, epochs_count

def plot_evaluation(accuracy, val_accuracy, loss, val_loss, f1_score, val_f1_score, epochs_count):
    # Plot evaluation metrics
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plt.plot(epochs_count, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs_count, val_accuracy, 'r', label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1)
    plt.xticks(epochs_count)
    plt.title('Training and validation accuracy')
    plt.legend()
        
    plt.subplot(3, 1, 2)
    plt.plot(epochs_count, loss, 'b', label='Training loss')
    plt.plot(epochs_count, val_loss, 'r', label='Validation loss')
    plt.ylabel('Loss')
    plt.ylim(0, 0.7)
    plt.xticks(epochs_count)
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs_count, f1_score[0], 'b', label='Training F1 Score')
    plt.plot(epochs_count, val_f1_score[0], 'r', label='Validation Score')
    plt.ylabel('F1 Score')
    plt.ylim(0.4, 1)
    plt.xticks(epochs_count)
    plt.title('Training and validation F1 score')
    plt.legend()

    plt.xlabel('Epochs')
    plt.show()
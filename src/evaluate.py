from matplotlib import pyplot as plt


def evaluate_model(history, epochs):   
    # Evaluate the model
    accuracy = history.history['binary_accuracy']
    val_accuracy = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, epochs + 1)      

    # Plot model evaluation
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1)
    plt.xticks(epochs)
    plt.title('Training and validation accuracy')
    plt.legend()
        
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.ylabel('Loss')
    plt.ylim(0, 0.7)
    plt.xticks(epochs)
    plt.xlabel('Epochs')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
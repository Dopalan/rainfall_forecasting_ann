from src.evaluate import evaluate_acc_loss, plot_evaluation
from src.model import build_model, repare_dataset, train_model


if __name__ == "__main__":
    # Prepare the dataset
    train_dataset, test_dataset, input_shape = repare_dataset('data/raw/weather.csv')
    
    # Build the model
    model = build_model(input_shape)
    
    # Epochs
    epochs = 10
    
    # Train the model
    history = train_model(model, train_dataset, test_dataset, epochs=epochs)
    
    # Evaluate the model
    accuracy, val_accuracy, loss, val_loss, f1_score, val_f1_score, epochs_count = evaluate_acc_loss(history, epochs)
    
    # Plot evaluation
    plot_evaluation(accuracy, val_accuracy, loss, val_loss, f1_score, val_f1_score, epochs_count)
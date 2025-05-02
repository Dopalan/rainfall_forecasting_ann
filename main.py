from src.evaluate import evaluate_model
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
    evaluate_model(history, epochs)
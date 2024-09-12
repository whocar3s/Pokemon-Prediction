from pathlib import Path
import pandas as pd
import pickle
from loguru import logger
import typer

from config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def load_model(model_path: Path):
    """Load a model from a pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "GB_model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info("Performing inference for model...")
    
    # Cargar datos de prueba
    try:
        logger.info(f"Loading test features from {features_path}...")
        features = pd.read_csv(features_path)
        logger.info("Test features loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load test features: {e}")
        return
    
    # Cargar el modelo
    try:
        logger.info(f"Loading model from {model_path}...")
        model = load_model(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Hacer predicciones
    try:
        logger.info("Making predictions...")
        predictions = model.predict(features)
        logger.info("Predictions made successfully.")
        
        # Guardar predicciones
        predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
    
    logger.success("Inference complete.")

if __name__ == "__main__":
    app()

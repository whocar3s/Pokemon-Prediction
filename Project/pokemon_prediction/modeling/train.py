# pokemon_prediction/modeling/train.py
import pandas as pd
import typer
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from loguru import logger
from tqdm import tqdm
from config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_dir: Path = MODELS_DIR,
):
    logger.info("Training models...")

    # Cargar datos
    try:
        logger.info("Loading features and labels...")
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        logger.info(f"Features and labels loaded from {features_path} and {labels_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)
    X_test.to_csv(PROCESSED_DATA_DIR / "test_features.csv", index=False)
    logger.info(f"Test features saved to {PROCESSED_DATA_DIR / 'test_features.csv'}")

    # Definir y entrenar modelos
    models = [
        ('LR', LinearRegression()),
        ('EN', ElasticNet()),
        ('Lasso', Lasso()),
        ('KNN', KNeighborsRegressor()),
        ('GB', GradientBoostingRegressor()),
        ('Ada', AdaBoostRegressor())
    ]

    for model_name, model in models:
        try:
            logger.info(f"Training {model_name} model...")
            model.fit(X_train, y_train)
            
            # Guardar el modelo
            model_path = model_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"{model_name} model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to train and save {model_name} model: {e}")

    logger.success("Model training complete.")

if __name__ == "__main__":
    app()

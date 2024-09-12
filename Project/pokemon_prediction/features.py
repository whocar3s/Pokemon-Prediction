from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info("Generating features from dataset...")

    # Leer datos procesados
    try:
        logger.info("Loading processed data...")
        data = pd.read_csv(input_path)
        logger.info(f"Data loaded from {input_path}")

        # Aquí puedes agregar más características o transformaciones si es necesario
        # Ejemplo: Generar estadísticas adicionales, realizar selecciones de características, etc.

        # Guardar el conjunto de datos de características
        data.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate features: {e}")

    logger.success("Feature generation complete.")

if __name__ == "__main__":
    app()

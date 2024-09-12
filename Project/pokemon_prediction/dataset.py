from pathlib import Path
import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    pokemon_input: Path = RAW_DATA_DIR / "pokemon.csv",
    combat_input: Path = RAW_DATA_DIR / "combats.csv",
    tests_input: Path = RAW_DATA_DIR / "tests.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
):
    logger.info("Processing dataset...")

    # Leer datos
    try:
        logger.info("Loading pokemon data...")
        pokemon_data = pd.read_csv(pokemon_input)
        logger.info(f"Pokemon data loaded from {pokemon_input}")

        logger.info("Loading combat data...")
        combat_data = pd.read_csv(combat_input)
        logger.info(f"Combat data loaded from {combat_input}")

        logger.info("Loading tests data...")
        tests_data = pd.read_csv(tests_input)
        logger.info(f"Tests data loaded from {tests_input}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Procesar datos
    logger.info("Starting data processing...")

    try:
        pokemon_data = pokemon_data.rename(columns = {'#':"ID"})

        FirstCombat = combat_data.First_pokemon.value_counts().reset_index(name='FirstCombat')
        SecondCombat = combat_data.Second_pokemon.value_counts().reset_index(name='SecondCombat')

        # Renombrar las columnas 'First_pokemon' y 'Second_pokemon' a 'pokemon_id'
        FirstCombat.rename(columns={'First_pokemon': 'pokemon_id'}, inplace=True)
        SecondCombat.rename(columns={'Second_pokemon': 'pokemon_id'}, inplace=True)


        # Fusionar ambos DataFrames en 'pokemon_id'
        TotalCombat = pd.merge(FirstCombat, SecondCombat, how='outer', on='pokemon_id').fillna(0)

        # Calcular el número total de combates
        TotalCombat['TotalMatch'] = TotalCombat['FirstCombat'] + TotalCombat['SecondCombat']

        # Contar las victorias cuando un Pokémon fue el primer o el segundo participante
        FirstWin = combat_data['First_pokemon'][combat_data['First_pokemon'] == combat_data['Winner']].value_counts().reset_index(name='FirstWin')
        SecondWin = combat_data['Second_pokemon'][combat_data['Second_pokemon'] == combat_data['Winner']].value_counts().reset_index(name='SecondWin')

        # Renombrar las columnas 'First_pokemon' y 'Second_pokemon' a 'pokemon_id'
        FirstWin.rename(columns={'First_pokemon': 'pokemon_id'}, inplace=True)
        SecondWin.rename(columns={'Second_pokemon': 'pokemon_id'}, inplace=True)

        #  Fusionar ambos DataFrames en 'pokemon_id'
        TotalWin = pd.merge(FirstWin, SecondWin, how='outer', on='pokemon_id').fillna(0)

        # Calcular el número total de victorias para cada Pokémon
        TotalWin['TotalWin'] = TotalWin['FirstWin'] + TotalWin['SecondWin']

        # Renombrar la columna 'index' en TotalCombat y TotalWin a 'pokemon_id'
        TotalCombat.rename(columns={'index': 'pokemon_id'}, inplace=True)
        TotalWin.rename(columns={'index': 'pokemon_id'}, inplace=True)

        # Realizar el merge entre pokemon y TotalCombat
        result = pd.merge(pokemon_data, TotalCombat, how='left', left_on='ID', right_on='pokemon_id')

        # Realizar el merge con TotalWin
        result = pd.merge(result, TotalWin, how='left', on='pokemon_id')

        # Eliminar la columna 'pokemon_id' si no la necesitas
        result = result.drop(['pokemon_id'], axis=1)

        # Calcular el porcentaje de victorias
        result['WinningPercentage'] = (result.TotalWin / result.TotalMatch) * 100

        result['Type 2'].fillna('Not Applicable', inplace = True)

        #eliminar las filas con valores nulos
        result = result.dropna()

        result.loc[result['Type 2'] != 'Not Applicable', 'Char'] = 'Both_Char'
        result.loc[result['Type 2'] == 'Not Applicable', 'Char'] = 'Only_One_Char'

        Scaleing_result = result

        from sklearn.preprocessing import StandardScaler

        col_name = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','FirstWin','SecondWin','TotalWin']
        scale = StandardScaler()
        Scaleing_result[col_name] = scale.fit_transform(Scaleing_result[col_name])

        Encoding_result = Scaleing_result.drop(['ID','Name','FirstCombat','SecondCombat','TotalMatch'],axis =1)
        Encoding_result['Legendary'] = Encoding_result['Legendary'].astype(str)
        Encoding_result = pd.get_dummies(Encoding_result, drop_first = True)

        WinningPercentage = Encoding_result['WinningPercentage']

        Encoding_result.drop(['WinningPercentage'], axis =1, inplace = True)

        # Seleccionar columnas para el modelo
        final_data = Encoding_result

        # Guardar los datos procesados
        final_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        # Guardar etiquetas
        labels = pd.DataFrame(WinningPercentage)
        labels.to_csv(labels_path, index=False)
        logger.info(f"Labels saved to {labels_path}")


    except Exception as e:
        logger.error(f"Failed to process and save data: {e}")

    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()

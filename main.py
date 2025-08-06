import multiprocessing as mp

import time
import pandas as pd
from pathlib import Path

import items_forecast

# from items_forecast import bronze, silver

# needs to be done before importing gold
items_forecast.bronze.bronze_pipeline()
items_forecast.silver.silver_pipeline()

from items_forecast import gold

months_ahead_to_forecast = 3

def wrap_get_predictions(item_numbers):

    result = gold.get_predictions(item_numbers, months_ahead_to_forecast)
    return result

fold_size = 8

all_item_numbers = gold.all_item_numbers[0:50]

if __name__ == "__main__":
    start = time.time()

    folds = [all_item_numbers[i:i + fold_size] for i in range(0, len(all_item_numbers), fold_size)]
    months_ahead_to_forecast = [3 for i in range(len(folds))]
    with mp.Pool(processes=14) as pool:
        results = pool.map(wrap_get_predictions, folds)

    combined_results = pd.concat(results)

    save_path = Path(__file__).parent / Path('items_forecast/GOLD/new_predictions.csv')

    combined_results.to_csv(save_path)

    end = time.time()
    print('DURATION: ', end - start)


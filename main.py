import multiprocessing as mp
import time
import pandas as pd
from pathlib import Path
from items_forecast import bronze_pipeline, silver_pipeline, gold_pipeline, get_all_item_numbers

# bronze_pipeline()
# silver_pipeline()

months_ahead_to_forecast = 3

def wrap_gold_pipeline(item_numbers):

    result = gold_pipeline(item_numbers, months_ahead_to_forecast)
    return result

num_folds = 14
all_item_numbers = get_all_item_numbers()

def split_into_n_folds(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

item_numbers = all_item_numbers[0:1000]

if __name__ == "__main__":
    start = time.time()

    folds = split_into_n_folds(item_numbers, num_folds)
    months_ahead_to_forecast = [3 for i in range(len(folds))]
    with mp.Pool(processes=14) as pool:
        results = pool.map(wrap_gold_pipeline, folds)

    combined_results = pd.concat(results)
    save_path = Path(__file__).parent / Path('items_forecast/GOLD/new_predictions.csv')
    combined_results.to_csv(save_path)

    end = time.time()
    print('DURATION: ', end - start)


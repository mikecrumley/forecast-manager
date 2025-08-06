
def bronze_pipeline():

    import pandas as pd
    import os
    from pathlib import Path

    raw_data_path = Path(__file__).parent / Path('raw_data/')

    orders_received = pd.read_csv(raw_data_path / Path('OrdersReceived_8-5-2025.csv'))
    june_order_fc = pd.read_csv(raw_data_path /Path('June order FC.csv'))
    item_remap = pd.read_csv(raw_data_path / Path('2024 Item remap table.csv'), encoding='latin1')

    bronze_path = Path(__file__).parent / Path('BRONZE')

    orders_received.to_csv(bronze_path / Path('orders_received.csv'), index=False)
    june_order_fc.to_csv(bronze_path / Path('June order FC.csv'),index=False)
    item_remap.to_csv(bronze_path / Path('2024 Item remap table.csv'),index=False)

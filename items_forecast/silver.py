
def silver_pipeline():
    import pandas as pd
    from pathlib import Path

    # load data from Bronze

    bronze_path = Path(__file__).parent / Path('BRONZE')
    orders_received = pd.read_csv(bronze_path / Path('orders_received.csv'))
    june_order_fc = pd.read_csv(bronze_path / Path('June order FC.csv'))
    item_remap = pd.read_csv(bronze_path / Path('2024 Item remap table.csv'))

    # prune data
    orders_received['ItemClass'] = orders_received['ItemClass'].str.strip()
    orders_received['ItemNumber'] = orders_received['ItemNumber'].str.strip()
    orders_received['BookingDate'] = pd.to_datetime(orders_received['BookingDate'])

    orders_received['Quantity'] = pd.to_numeric(orders_received['Quantity'], errors='coerce')
    orders_received['BookingDate'] = pd.to_datetime(orders_received['BookingDate'],errors='raise')
    orders_received['Month'] = orders_received['BookingDate'].dt.to_period('M')

    # date range ends on the first of this month
    end_date = pd.Timestamp.today().replace(day=1).normalize()

    # go five years back
    start_date = (end_date - pd.DateOffset(years=5)).normalize()

    # Filter orders between first of THIS month and five years earlier
    # This will be our sample data from now on
    orders_received = orders_received[
        (orders_received['BookingDate'] > start_date) &
        (orders_received['BookingDate'] < end_date)
    ]

    silver_path = Path(__file__).parent / Path('SILVER')

    orders_received.to_csv(silver_path / Path('orders_received.csv'))
    june_order_fc.to_csv(silver_path / Path('June order FC.csv'))
    item_remap.to_csv(silver_path / Path('2024 Item remap table.csv'))

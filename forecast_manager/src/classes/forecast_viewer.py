import pandas as pd
import random as rand
import matplotlib.pyplot as plt

class ForecastViewer:

    def __init__(self, orders_received_df, predictions_df):

        self.orders_received_df = orders_received_df.copy()
        self.predictions_df = predictions_df.copy()

        self.predictions_df.index = self.predictions_df['ItemNumber']
        del self.predictions_df['ItemNumber']
        # del self.predictions_df['forecaster_class']
        # del self.predictions_df['forecaster_params']
        # del self.predictions_df['test_RMSE']

        self.predictions_df.columns = pd.PeriodIndex(self.predictions_df.columns, freq="M")

        # Convert the column in-place
        self.orders_received_df['BookingDate'] = pd.to_datetime(
            self.orders_received_df['BookingDate'],
            errors='raise'  # will throw if any non-parseable entries exist
        )

        self.monthly_item_sales_quantity = pd.pivot_table(
            self.orders_received_df,
            index='ItemNumber',
            columns=self.orders_received_df['BookingDate'].dt.to_period('M'),
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )

        self.monthly_item_sales_quantity.index.name = 'ItemNumber'
        self.item_numbers = list(self.predictions_df.index)

    @staticmethod
    def _to_datetime_index(s: pd.Series) -> pd.Series:
        """Convert PeriodIndex → DatetimeIndex; leave others as-is."""
        if isinstance(s.index, pd.PeriodIndex):
            # choose 'start' or 'end' anchor to taste for monthly/quarterly data
            s = s.copy()
            s.index = s.index.to_timestamp(how="start")

        return s

    def plot(self, item_number = None):

        if item_number is None:
            item_number = rand.choice(self.item_numbers)

        historical_series = self.monthly_item_sales_quantity.loc[item_number]
        forecast_series = self.predictions_df.loc[item_number]

        hist = self._to_datetime_index(historical_series).sort_index()
        fcst = self._to_datetime_index(forecast_series).sort_index()

        # Always create our own figure/axes here
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(hist.index, hist.values, label="History")
        ax.plot(fcst.index, fcst.values, linestyle="--", label="Forecast")

        # Optional: mark boundary between history and forecast
        if len(hist) and len(fcst):
            ax.axvline(hist.index.max(), color="gray", alpha=0.3)

        ax.set_title(item_number)
        ax.set_ylabel('quantity')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

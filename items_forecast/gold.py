import pandas as pd
from pathlib import Path
import multiprocessing as mp
import matplotlib.pyplot as plt

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter, temporal_train_test_split
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_X, check_y
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tqdm import tqdm

from forecast_manager import ForecasterEvaluation, \
    ForecastResult, \
    ManagedForecaster, \
    ManagedCrossValidator, \
    ManagedMetric, \
    ManagedEvaluator, \
    ForecastManager

def get_all_item_numbers():
    silver_path = Path(__file__).parent / Path('SILVER')

    orders_received = pd.read_csv(silver_path / Path('orders_received.csv'))
    june_order_fc = pd.read_csv(silver_path / Path('June order FC.csv'))
    item_remap = pd.read_csv(silver_path / Path('2024 Item remap table.csv'))

    # Convert the column in-place
    orders_received['BookingDate'] = pd.to_datetime(
        orders_received['BookingDate'],
        errors='raise'  # will throw if any non-parseable entries exist
    )

    # build monthly quantity series for all item_numbers

    monthly_item_sales_quantity = pd.pivot_table(
        orders_received,
        index='ItemNumber',
        columns=orders_received['BookingDate'].dt.to_period('M'),
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    monthly_item_sales_quantity.index.name = 'ItemNumber'

    all_item_numbers = list(monthly_item_sales_quantity.index)
    all_item_numbers.sort()

    return all_item_numbers



def gold_pipeline(item_numbers, months_ahead_to_forecast = 3):
    # load data from Silver
    silver_path = Path(__file__).parent / Path('SILVER')

    orders_received = pd.read_csv(silver_path / Path('orders_received.csv'))
    june_order_fc = pd.read_csv(silver_path / Path('June order FC.csv'))
    item_remap = pd.read_csv(silver_path / Path('2024 Item remap table.csv'))

    # Convert the column in-place
    orders_received['BookingDate'] = pd.to_datetime(
        orders_received['BookingDate'],
        errors='raise'    # will throw if any non-parseable entries exist
    )

    # build monthly quantity series for all item_numbers

    monthly_item_sales_quantity = pd.pivot_table(
        orders_received,
        index='ItemNumber',
        columns=orders_received['BookingDate'].dt.to_period('M'),
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    monthly_item_sales_quantity.index.name = 'ItemNumber'

    all_item_numbers = list(monthly_item_sales_quantity.index)
    all_item_numbers.sort()


    # from Joe's spreadsheet
    item_classes = [
        "CART", "CHOC", "CPU", "CRAN", "DEIC", "EGST", "ENCM", "ESRV", "FSRV",
        "FUEL", "HPU", "HTAR", "JACK", "JP", "KIT", "LIFT", "LSRV", "MISC",
        "NOIN", "NSRV", "OSRV", "PART", "PLAT", "PWTR", "RAM", "RMA", "SAIR",
        "SLNG", "SPCH", "STND", "SVC", "TAIL", "TBAR", "TIRE", "TOOL", "TUG", "SVCP",
        "", "JP"
    ]

    segment_descriptions = [
        "Kits & Parts",
        "Jacks & Stands",
        "Hydraulic Systems",
        "Other",
        "Tugs",
        "Servicing Products",
        "Towbars",
        "JetPorter",
        "RAM Air Turbines",
        "Electrical Systems",
        "N2O2 Systems",
        "Engine Comp Washers",
        "Work Platforms"
    ]

    # map from segment descriptions in Joe's spreadsheet to ItemClasses in orders_received
    segment_to_item_classes = {
        'Other': [
            'CART', 'CHOC', 'CPU', 'CRAN', 'DEIC', 'HTAR', 'LIFT', 'MISC',
            'NOIN', 'PWTR', 'RMA', 'SAIR', 'SLNG', 'SPCH', 'STND', 'SVC',
            'TOOL', 'SVCP', '', "JP"
        ],
        'Engine Comp Washers': ['ENCM'],
        'Electrical Systems': ['ESRV'],
        'Servicing Products': ['FSRV', 'FUEL', 'LSRV', 'TIRE'],
        'Hydraulic Systems': ['HPU'],
        'Jacks & Stands': ['JACK', 'TAIL'],
        'Kits & Parts': ['KIT', 'PART'],
        'N2O2 Systems': ['NSRV', 'OSRV'],
        'Work Platforms': ['PLAT'],
        'RAM Air Turbines': ['RAM'],
        'Towbars': ['TBAR'],
        'Tugs': ['TUG']
    }

    forecast_specs = pd.DataFrame(index = all_item_numbers, columns = ['forecaster_class','forecaster_params', 'test_RMSE'])


    start = pd.Timestamp.today().to_period('M')

    # forecast horizon so many months in the future
    fh = pd.period_range(start=start, periods=months_ahead_to_forecast, freq='M')

    predictions = pd.DataFrame(index = item_numbers, columns = fh)

    count = -1

    for item_number in item_numbers:
        count += 1

        ts = monthly_item_sales_quantity.loc[item_number]

        ############# forecast configs ###################

        forecaster_class_1 = NaiveForecaster
        forecaster_class_2 = NaiveForecaster
        forecaster_class_3 = NaiveForecaster
        forecaster_class_4 = TransformedTargetForecaster
        forecaster_class_5 = ExponentialSmoothing           # Holt-Winters
        forecaster_class_6 = AutoARIMA


        forecaster_params_1 = {
            'strategy':'last'
        }

        forecaster_params_2 = {
            'strategy':'mean',
            'window_length': 3
        }

        forecaster_params_3 = {
            'strategy':'mean',
            'window_length': 12
        }

        forecaster_params_4 = {
            'steps':[("deseasonalize", Deseasonalizer(model="additive", sp=12)),
            ("trend", PolynomialTrendForecaster(degree=1))]
        }

        forecaster_params_5 = {
            'trend': "add",
            'seasonal': "add",
            'sp': 12
        }

        forecaster_params_6 = {}      #{'order':(1,1,1),'seasonal_order':(1,1,1)}

        splitter_class_1 = SlidingWindowSplitter

        cv_params_1 = {'window_length':24, 'step_length': 12, 'fh': range(1,months_ahead_to_forecast+1)}

        metric_1 = MeanSquaredError(square_root=True)

        ##############################################################

        forecaster_1 = ManagedForecaster(forecaster_class=forecaster_class_1,init_params=forecaster_params_1)
        forecaster_2 = ManagedForecaster(forecaster_class=forecaster_class_2,init_params=forecaster_params_2)
        forecaster_3 = ManagedForecaster(forecaster_class=forecaster_class_3,init_params=forecaster_params_3)
        forecaster_4 = ManagedForecaster(forecaster_class=forecaster_class_4,init_params=forecaster_params_4)
        forecaster_5 = ManagedForecaster(forecaster_class=forecaster_class_5,init_params=forecaster_params_5)
        forecaster_6 = ManagedForecaster(forecaster_class=forecaster_class_6,init_params=forecaster_params_6)

        managed_cv_1 = ManagedCrossValidator(splitter_class=splitter_class_1, init_params=cv_params_1)

        managed_metric_1 = ManagedMetric(metric_1)

        managed_evaluator_1 = ManagedEvaluator(
            managed_cv = managed_cv_1,
            strategy="refit",
            scoring=managed_metric_1,
            error_score='raise')

        fm = ForecastManager(y=ts, name=item_number)

        fm.add_forecaster(forecaster_1)
        fm.add_forecaster(forecaster_2)
        fm.add_forecaster(forecaster_3)
        fm.add_forecaster(forecaster_4)
        fm.add_forecaster(forecaster_5)
        fm.add_forecaster(forecaster_6)

        fm.add_evaluator(managed_evaluator_1)

        #################################################


        fm.evaluate()
        summary = fm.summarize_evaluations()

        third_col = summary.columns[2]

        idx = summary[third_col].idxmin()

        winning_forecaster_name = summary.loc[idx].name

        fm.fit_forecaster(winning_forecaster_name)

        forecast_result=fm.predict_forecaster(winning_forecaster_name, fh)
        forecast_series = forecast_result.results

        predictions.loc[item_number] = forecast_series
        forecast_specs.loc[item_number] = list(summary.loc[idx])

    result = pd.merge(predictions, forecast_specs, left_index=True, right_index=True)

    return result



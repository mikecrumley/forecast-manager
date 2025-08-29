from sktime.forecasting.model_evaluation import evaluate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random as rand

class ForecasterEvaluation:
    def __init__(self, results, managed_evaluator):
        self.results = results
        self.managed_evaluator = managed_evaluator


class ForecastResult:
    def __init__(self, fh, results, forecaster):
        self.fh = fh  # Forecast horizon used for the prediction
        self.results = results  # Forecasted values (typically a pandas Series)
        self.forecaster = forecaster  # Reference to the forecaster that produced this result


class ManagedForecaster:
    def __init__(self, forecaster_class, init_params=None, name=None):
        self.init_params = init_params or {}  # Parameters used to instantiate the model
        self.name = name or f"{forecaster_class.__name__}({self.init_params})"  # Unique identifier for this forecast instance
        self.forecaster = forecaster_class(**self.init_params)  # Instantiate forecaster
        self.is_fitted = False
        self.forecaster_evaluations = {}  # Changed from list to dict
        self.forecast_results = []
        self.fitted_params = None

    def fit(self, y, X=None, fh=None):
        if X is None or not self.forecaster.get_tag("X_inner_mtype"):
            self.forecaster.fit(y)
        else:
            self.forecaster.fit(y, X, fh)
        self.is_fitted = True
        try:
            self.fitted_params = self.forecaster.get_fitted_params()
        except Exception:
            self.fitted_params = None

    def evaluate(self, y, X=None, managed_evaluator=None):

        # here is where we should catch any models that fail to evaluate.
        # we return an evaluation that disqualifies it from
        # consideration for the best model
        try:
            results = evaluate(
                forecaster=self.forecaster,
                y=y,
                X=X,
                cv=managed_evaluator.managed_cv.cv,
                strategy=managed_evaluator.strategy,
                scoring=[m.metric for m in managed_evaluator.scoring],
                error_score=managed_evaluator.error_score
            )
            evaluation = ForecasterEvaluation(results, managed_evaluator)

        except:
            # If this fails, ForecastManager.summarize_evaluations will
            # ignore this None value
            evaluation = None

        self.forecaster_evaluations[managed_evaluator.managed_cv.name] = evaluation

    def predict(self, fh, X_test=None):
        if not self.is_fitted:
            raise RuntimeError(f"Forecaster '{self.name}' must be fitted before prediction.")
        if X_test is None or not self.forecaster.get_tag("X_inner_mtype"):
            results = self.forecaster.predict(fh)
        else:
            results = self.forecaster.predict(fh, X_test)
        forecast_result = ForecastResult(fh, results, self)
        self.forecast_results.append(forecast_result)
        return forecast_result


class ManagedCrossValidator:
    def __init__(self, splitter_class, init_params=None, name=None):
        self.splitter_class = splitter_class
        self.init_params = init_params or {}
        self.cv = splitter_class(**self.init_params)
        self.name = name or f"{splitter_class.__name__}({self.init_params})"


class ManagedMetric:
    def __init__(self, metric):
        self.metric = metric
        cls_name = metric.__class__.__name__
        params = metric.get_params()
        default_params = type(metric)().get_params()
        non_default = {k: v for k, v in params.items() if default_params.get(k) != v}
        if non_default:
            param_str = ", ".join(f"{k}={v}" for k, v in non_default.items())
            self.name = f"{cls_name}({param_str})"
        else:
            self.name = cls_name


class ManagedEvaluator:
    def __init__(self, managed_cv, strategy="refit", scoring=None, error_score="raise"):
        self.managed_cv = managed_cv
        self.strategy = strategy
        if isinstance(scoring, ManagedMetric):
            self.scoring = [scoring]
        elif isinstance(scoring, list) and all(isinstance(s, ManagedMetric) for s in scoring):
            self.scoring = scoring
        else:
            raise ValueError("scoring must be a ManagedMetric or a list of ManagedMetric objects")
        self.error_score = error_score


class ForecastManager:

    def __init__(self, y, X=None, name=None):
        self.y = y
        self.X = X
        self.name = name
        self.managed_forecasters = []
        self.managed_evaluators = []
        self._name_counter = 0

    def add_forecaster(self, managed_forecaster):
        if any(f.name == managed_forecaster.name for f in self.managed_forecasters):
            raise ValueError(f"A forecaster with name '{managed_forecaster.name}' already exists.")
        self.managed_forecasters.append(managed_forecaster)

    def add_evaluator(self, managed_evaluator):
        if any(e is managed_evaluator for e in self.managed_evaluators):
            raise ValueError("This evaluator instance is already added.")
        self.managed_evaluators.append(managed_evaluator)

    def evaluate_forecaster(self, name, managed_evaluator):
        matched = [f for f in self.managed_forecasters if f.name == name]
        if not matched:
            raise ValueError(f"No forecaster found with name '{name}'.")
        managed_forecaster = matched[0]
        return managed_forecaster.evaluate(y=self.y, X=self.X, managed_evaluator=managed_evaluator)

    def evaluate(self):
        total = len(self.managed_forecasters) * len(self.managed_evaluators)
        with tqdm(total=total, desc="Evaluating forecasters") as pbar:
            for forecaster in self.managed_forecasters:
                for evaluator in self.managed_evaluators:
                    self.evaluate_forecaster(forecaster.name, evaluator)
                    pbar.update(1)

    def predict_forecaster(self, name, fh, X_test=None):
        matched = [f for f in self.managed_forecasters if f.name == name]
        if not matched:
            raise ValueError(f"No forecaster found with name '{name}'.")
        managed_forecaster = matched[0]
        return managed_forecaster.predict(fh, X_test)

    def fit_forecaster(self, name, fh=None):
        matched = [f for f in self.managed_forecasters if f.name == name]
        if not matched:
            raise ValueError(f"No forecaster found with name '{name}'.")
        managed_forecaster = matched[0]
        return managed_forecaster.fit(self.y, self.X, fh)

    def plot_series(self):
        plt.figure(figsize=(10, 5))
        y_plot = self.y.copy()
        if isinstance(y_plot.index, pd.PeriodIndex):  # Convert PeriodIndex to Timestamp for plotting compatibility
            y_plot.index = y_plot.index.to_timestamp()
        plt.plot(y_plot, label="Target Series")
        plt.title(f"{self.name if self.name else 'ForecastManager'}: Target Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_exogeneous_series(self):
        if self.X is None or self.X.empty:
            print("No exogenous series to plot.")
            return

        plt.figure(figsize=(10, 5))
        X_plot = self.X.copy()
        if isinstance(X_plot.index, pd.PeriodIndex):  # Convert PeriodIndex to Timestamp for plotting compatibility
            X_plot.index = X_plot.index.to_timestamp()
        for col in X_plot.columns:
            plt.plot(X_plot.index, X_plot[col], label=col)
        plt.title(f"{self.name if self.name else 'ForecastManager'}: Exogenous Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_forecasts(self, *forecast_results, plot_series=False):
        plt.figure(figsize=(12, 6))
        if plot_series:
            y_plot = self.y.copy()
            if isinstance(y_plot.index, pd.PeriodIndex):  # Convert PeriodIndex to Timestamp for plotting compatibility
                y_plot.index = y_plot.index.to_timestamp()
            plt.plot(y_plot, label="Actual", linestyle="--", color="gray")

        for i, fr in enumerate(forecast_results):
            fr_plot = fr.results.copy()
            if isinstance(fr_plot.index, pd.PeriodIndex):  # Convert PeriodIndex to Timestamp for plotting compatibility
                fr_plot.index = fr_plot.index.to_timestamp()
            plt.plot(fr_plot, label=fr.forecaster.name)

        plt.title(f"{self.name if self.name else 'ForecastManager'} Forecast Results")
        plt.xlabel("Time")
        plt.ylabel("Forecasted Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_evaluator(self, managed_evaluator):
        cv = managed_evaluator.managed_cv.cv
        splits = list(cv.split(self.y))

        plt.figure(figsize=(12, 4))
        for i, (train_idx, test_idx) in enumerate(splits):
            plt.plot(train_idx, [i]*len(train_idx), 'b|-', label='Train' if i == 0 else "")
            plt.plot(test_idx, [i]*len(test_idx), 'ro', label='Test (fh)' if i == 0 else "")

        plt.xlabel("Time Index")
        plt.ylabel("Split")
        plt.yticks(range(len(splits)), [f"Split {i+1}" for i in range(len(splits))])
        title = f"{managed_evaluator.managed_cv.name}"
        plt.title(f"Visualization of {title}")
        plt.legend()

        if managed_evaluator.scoring is not None:
            metrics_text = "Scoring: "
            metrics_text += ", ".join(m.name for m in managed_evaluator.scoring)
            plt.gcf().text(0.99, 0.01, metrics_text, fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_evaluators(self):
        for evaluator in self.managed_evaluators:
            self.plot_evaluator(evaluator)

    def summarize_evaluations(self):
        records = []
        index = []
        column_keys = set()

        exclude_columns = {'fit_time', 'len_train_window', 'pred_time'}

        for forecaster in self.managed_forecasters:
            print('forecaster: ', forecaster)
            row = {
                'forecaster_name': forecaster.name
                # 'forecaster_class': forecaster.forecaster.__class__.__name__,
                # 'init_params': forecaster.init_params
            }
            metrics = {}
            for evaluator in self.managed_evaluators:
                cv_name = evaluator.managed_cv.name
                eval_obj = forecaster.forecaster_evaluations.get(cv_name)
                if eval_obj is not None:
                    df = eval_obj.results
                    scores = df.mean(numeric_only=True).drop(labels=exclude_columns, errors='ignore').to_dict()
                    for metric, value in scores.items():
                        key = (cv_name, metric)
                        metrics[key] = value
                        column_keys.add(key)
            row.update(metrics)
            records.append(row)
            index.append(forecaster.name)

        all_columns = ['forecaster_name'] + sorted(column_keys)
        df = pd.DataFrame(records, index=index)
        df.columns = pd.MultiIndex.from_tuples(all_columns)
        return df

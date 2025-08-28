

class ForecastPortfolio:
    '''A class to perform automatic model selection and forecasting
    for a large number of time series and forecasters.
    Also wraps it in logic to run in spark.'''

    def __init__(self,
                 series_df,
                 managed_forecaster_mapping,
                 managed_evaluator_mapping,
                 weights_series,  # weights to use for portfolio scoring
                 forecast_horizon,
                 exogenous_df_mapping=lambda idx: None):

        self.series_df = series_df
        self.managed_forecaster_mapping = managed_forecaster_mapping
        self.exogeneous_df_mapping = exogenous_df_mapping
        self.managed_evaluator_mapping = managed_evaluator_mapping
        self.weights_series = weights_series
        self.forecast_horizon = forecast_horizon

        self.forecast_manager_mapping = {}

        self.evaluation_results = None
        self.predictions = None

        # build individual ForecastManager instances
        for idx in self.series_df.index:
            exogeneous_df = exogenous_df_mapping(idx)
            fm = ForecastManager(y=series_df.loc[idx], X=exogeneous_df, name=idx)
            managed_forecasters = self.managed_forecaster_mapping(idx)
            managed_evaluator = self.managed_evaluator_mapping(idx)
            for mf in managed_forecasters:
                fm.add_forecaster(mf)
            fm.add_evaluator(managed_evaluator)
            self.forecast_manager_mapping[idx] = fm

    def _evaluate_batch(self, indices):

        S = pd.Series(indices)

        def apply_func(idx):
            fm = self.forecast_manager_mapping[idx]
            fm.evaluate()
            summary_df = fm.summarize_evaluations()
            winning_row = summary_df.loc[summary_df[summary_df.columns[1]].idxmin()]
            winning_forecaster_name = winning_row.iloc[0]
            winning_cv_score = winning_row.iloc[1]
            new_s = pd.Series([winning_forecaster_name, winning_cv_score])
            new_s.name = 'index'

            return new_s

        # truncated_series_df = self.series_df.loc[indices]
        return_df = S.apply(apply_func)
        return_df.index = indices
        return_df.columns = ['forecaster_name', 'cv_score']

        return return_df

    def _predict_batch(self, indices):

        S = pd.Series(indices)

        def get_predictions_from_series(idx):
            s = self.evaluation_results.loc[idx]
            winning_forecaster_name = s['forecaster_name']
            fm = self.forecast_manager_mapping[idx]
            fm.fit_forecaster(winning_forecaster_name)
            prediction = fm.predict_forecaster(name=winning_forecaster_name, fh=self.forecast_horizon,
                                               X_test=self.exogeneous_df_mapping(idx))
            return_df = prediction.results
            return_df['forecaster_name'] = winning_forecaster_name
            return_df['cv_score'] = s['cv_score']
            return_df.name = idx
            return return_df

        return_df = S.apply(get_predictions_from_series)
        return_df.index = indices

        return return_df

    def _sanitize_for_spark(self, pdf):
        # 1) make sure all column NAMES are strings
        pdf = pdf.copy()
        pdf.columns = pdf.columns.map(str)

        # 2) convert Period-dtype columns to string values
        for c in pdf.columns:
            if is_period_dtype(pdf[c]):
                pdf[c] = pdf[c].astype(str)

        # (optional) if you know specific columns that are Period objects in values, cast them too
        return pdf

    def _run_evaluation_batch_for_spark(self, iterator):
        # Spark calls this once per partition; 'iterator' yields your batch lists
        for batch in iterator:
            pdf = self._evaluate_batch(batch)
            pdf['index'] = pdf.index
            pdf = self._sanitize_for_spark(pdf)
            # stream out rows; Row(**dict) guarantees string field names
            for rec in pdf.to_dict(orient="records"):
                yield Row(**rec)

    def _run_prediction_batch_for_spark(self, iterator):
        # Spark calls this once per partition; 'iterator' yields your batch lists
        for batch in iterator:
            pdf = self._predict_batch(batch)
            pdf['index'] = pdf.index
            pdf = self._sanitize_for_spark(pdf)
            for rec in pdf.to_dict(orient="records"):
                yield Row(**rec)

    def _create_batches_for_spark(self, num_batches=3 * 48):
        '''num_batches should usually be (2 - 4) x num_cores.
        Each batch corresponds to one spark task.'''

        all_indices = self.series_df.index
        num_indices = len(all_indices)

        batch_size = max(1, num_indices // num_batches)
        batches = [all_indices[i:i + batch_size] for i in range(0, num_indices, batch_size)]

        return batches

    def evaluate_in_pandas(self, save_path=None):
        '''skip spark, just do it in pandas.  for small batches.'''
        evaluation_results = self._evaluate_batch(indices=self.series_df.index)
        # evaluation_results.index = evaluation_results['index']
        # del evaluation_results['index']
        self.evaluation_results = evaluation_results
        return evaluation_results

    def evaluate_in_spark(self, num_batches=3 * 48, save_path=None, spark=None):
        '''Evaluate all time series in spark, store results as a dataframe'''

        if spark is None:
            spark = SparkSession.builder.getOrCreate()

        batches = self._create_batches_for_spark(num_batches=num_batches)
        rdd = spark.sparkContext.parallelize(batches, numSlices=len(batches))

        results_df = rdd.mapPartitions(self._run_evaluation_batch_for_spark).toDF()

        if save_path is not None:
            results_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(save_path)
            print(f'results written to {save_path}')

        evaluation_results = results_df.toPandas()
        evaluation_results.index = evaluation_results['index']
        del evaluation_results['index']
        self.evaluation_results = evaluation_results
        return evaluation_results

    def predict_in_pandas(self):
        '''Need to have evaluation results to do forecasts.'''
        if self.evaluation_results is None:
            raise RuntimeError(
                'Error in ForecastPortfolio.predict, self.evaluation_results is None.  Please perform evaluation first.')

        def get_prediction_from_series(s):
            idx = s.name
            s = self.evaluation_results.loc[idx]
            winning_forecaster_name = s['forecaster_name']
            fm = self.forecast_manager_mapping[idx]
            fm.fit_forecaster(winning_forecaster_name)
            prediction = fm.predict_forecaster(name=winning_forecaster_name, fh=self.forecast_horizon,
                                               X_test=self.exogeneous_df_mapping(idx))
            return_df = prediction.results
            return_df['forecaster_name'] = winning_forecaster_name
            return_df['cv_score'] = s['cv_score']
            return return_df

        predictions_df = self.evaluation_results.apply(get_prediction_from_series, axis=1)
        predictions_df.index.name = 'index'
        self.predictions = predictions_df
        return predictions_df

    def predict_in_spark(self, num_batches=3 * 48, save_path=None, spark=None):

        if spark is None:
            spark = SparkSession.builder.getOrCreate()

        batches = self._create_batches_for_spark(num_batches=num_batches)
        rdd = spark.sparkContext.parallelize(batches, numSlices=len(batches))

        results_df = rdd.mapPartitions(self._run_prediction_batch_for_spark).toDF()

        if save_path is not None:
            results_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(save_path)
            print(f'predictions written to {save_path}')

        predictions_df = results_df.toPandas()
        predictions_df.index = predictions_df['index']
        del predictions_df['index']

        self.predictions = predictions_df

        return predictions_df

    def score_portfolio(self, evaluation_results):
        # map from indices to floats; usually net sales over all orders_received
        weights_series = self.weights_series
        total_weight = weights_series.sum()
        portfolio_score = (evaluation_results[
                               'cv_score'].sort_index() * weights_series.sort_index()).sum() / total_weight
        return portfolio_score

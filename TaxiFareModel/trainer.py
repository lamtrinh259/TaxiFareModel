from sklearn.pipeline import Pipeline, make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

# Indicate mlflow to log to remote server
# mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
# client = MlflowClient()
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[JP] [Tokyo] Lam's unique experiment"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[JP] [Tokyo] Lam's unique experiment"
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    # For ML Flow
    # @memoized_property
    # def mlflow_client(self):
        # mlflow.set_tracking_uri(self.MLFLOW_URI)
        # return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data(nrows=1000)
    # clean data
    df = clean_data(df)
    # set X and y
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    print('RMSE is', rmse)

    # Establish the MLflow to track experiments
    # Set tracking location
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Client to track ML model
    client = MlflowClient()

    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    client.create_run(experiment_id)
    print(type(client))
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    # This doesn't work
    # experiment_id = trainer.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).mlflow_client.experiment_id

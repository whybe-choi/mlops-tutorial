import uuid
import mlflow
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

UNIQUE_PREFIX = str(uuid.uuid4())[:8]

def objective(trial):
    #
    # suggest new parameter
    #
    trial.suggest_int("n_estimators", 100, 1000, step=100)
    trial.suggest_int("max_depth", 3, 10)
    run_name = f"{UNIQUE_PREFIX}-{trial.number}"
    with mlflow.start_run(run_name=run_name):
        #
        # set parameters
        #
        mlflow.log_params(trial.params)

        #
        # load data
        #
        iris = load_iris(as_frame=True)
        X, y = iris['data'], iris['target']
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2024)

        #
        # train model
        #
        clf = RandomForestClassifier(n_estimators=trial.params["n_estimators"], max_depth=trial.params["max_depth"], random_state=2024)
        clf.fit(X_train, y_train)

        #
        # evaluate train model
        #
        y_pred = clf.predict(X_valid)
        acc_score = accuracy_score(y_pred, y_valid)

        #
        # log metric
        #
        mlflow.log_metric("accuracy", acc_score)
    return acc_score

if __name__ == "__main__":
    #
    # set mlflow
    #
    study_name = "hpo-tutorial"
    mlflow.set_tracking_uri("http://0.0.0.0:5001")
    mlflow.set_experiment(study_name)

    # study
    sampler = optuna.samplers.RandomSampler(seed=2024)
    study = optuna.create_study(sampler=sampler, study_name=study_name, direction="maximize")

    #optimize
    study.optimize(objective, n_trials=5)
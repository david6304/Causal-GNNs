from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor

class XGBBaseline(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=6, lr=0.1, num_classes=2, early_stopping_rounds=10, task="classification"):
        self.num_classes = num_classes
        self.task = task
        
        if task == "regression":
            self.estimator = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds
            )
            self.estimator = MultiOutputRegressor(self.estimator)
        else:
            if self.num_classes > 2:
                # Multi-class 
                self.estimator = XGBClassifier(
                    objective="multi:softprob",
                    num_class=self.num_classes,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    eval_metric="mlogloss",
                    early_stopping_rounds=early_stopping_rounds
                )
            else:
                # Binary
                self.estimator = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    eval_metric="logloss",
                    early_stopping_rounds=early_stopping_rounds
                )
    
    def __repr__(self):
        return "XGBBaseline"
    
    def fit(self, X, y, eval_set=None):
        """Trains the XGBoost model with an optional validation set."""
        self.estimator.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

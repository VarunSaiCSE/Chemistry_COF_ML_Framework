from dataclasses import dataclass
from typing import Dict, Any, Tuple


from xgboost import XGBRegressor
from catboost import CatBoostRegressor


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from configs.base_config import BaseConfig

@dataclass
class BaselineResults:
    r2: float
    rmse: float
    mae: float

class RandomForestBaseline:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            random_state=self.config.training.random_seed,
            n_jobs=-1,
        )

    def train_and_evaluate(self, X, y) -> BaselineResults:
        cfg = self.config
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_seed,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        print(f"📊 RandomForest Results:")
        print(f"   R²   : {r2:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")

        return BaselineResults(r2=r2, rmse=rmse, mae=mae)
    
class XGBoostBaseline:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = XGBRegressor(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.training.random_seed,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    def train_and_evaluate(self, X, y) -> BaselineResults:
        cfg = self.config
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_seed,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        print(f"📊 XGBoost Results:")
        print(f"   R²   : {r2:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")

        return BaselineResults(r2=r2, rmse=rmse, mae=mae)
    
class CatBoostBaseline:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = CatBoostRegressor(
            depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            iterations=self.config.model.n_estimators,
            loss_function="RMSE",
            random_seed=self.config.training.random_seed,
            verbose=False,
        )

    def train_and_evaluate(self, X, y) -> BaselineResults:
        cfg = self.config
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_seed,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        print(f"📊 CatBoost Results:")
        print(f"   R²   : {r2:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")

        return BaselineResults(r2=r2, rmse=rmse, mae=mae)
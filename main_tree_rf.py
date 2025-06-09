import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNRegressor,
)
from tabpfn import TabPFNRegressor
import os
import logging


def rank_cross_section_features(X: pd.DataFrame):
    """Simple Cross-Section Ranking [-0.5, 0.5]"""
    assert isinstance(X, pd.DataFrame)
    columns = X.columns

    def _rank(df: pd.DataFrame) -> pd.DataFrame:
        ranked_df = df[columns].rank(pct=True) - 0.5
        return ranked_df

    X = X.groupby("eom").apply(_rank, include_groups=False)
    return X


def random_fourier_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame, number_features: int, seed: int
):
    """Clean Random Fourier Feature, see Rahimi 2007."""
    np.random.seed(seed=int((seed + 1) * 1e3))
    weights = np.random.normal(
        loc=0, scale=1, size=[x_train.shape[1], int(number_features / 2)]
    )

    # np.__version__ == 1.26.4
    # XXX np.arange(0.5, 1.1, step=0.1) bugged :(
    gammas = np.random.choice(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        size=[x_train.shape[1], int(number_features / 2)],
        replace=True,
    )
    weights = weights * gammas
    x_train_rff = pd.concat(
        [getattr(np, activation)(x_train @ weights) for activation in ["cos", "sin"]],
        axis=1,
    )

    x_test_rff = pd.concat(
        [getattr(np, activation)(x_test @ weights) for activation in ["cos", "sin"]],
        axis=1,
    )
    return x_train_rff, x_test_rff


def ranked_random_fourier_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    number_features: int,
    seed: int,
):
    """Applies Ranking After RFF"""

    x_train_rff, x_test_rff = random_fourier_features(
        x_train=x_train,
        x_test=x_test,
        number_features=number_features,
        seed=seed,
    )
    x_train_rff.columns = np.arange(0, number_features, step=1)
    x_train_rff = rank_cross_section_features(x_train_rff)
    # XXX Monthly re-training
    x_test_rff = x_test_rff.rank(pct=True) - 0.5
    return x_train_rff, x_test_rff


logging.basicConfig(level=logging.INFO)
output_path = "results.csv"
df = pd.read_pickle("jkp_mega_only.pickle")
df = df[~df.ret_exc_lead1m.isna()]
df.pop("size_grp")

rolling_window = 60
dates = df.reset_index().eom.unique().tolist()
dates.sort()
test_dates = dates[rolling_window:]
portfolio_returns = []
use_rf = True
seed = 1234
number_features = 600

for test_date in tqdm(test_dates):
    start_time = time.time()
    start_index = dates.index(test_date) - rolling_window
    end_index = dates.index(test_date)
    in_sample_dates = dates[start_index:end_index]
    train = df[df.eom.isin(in_sample_dates)].copy()
    test: pd.DataFrame = df[df.eom == test_date].copy()

    train.set_index("eom", inplace=True)
    test.set_index("eom", inplace=True)
    train.pop("id")

    test_ids = test.pop("id")

    y_train = train.pop("ret_exc_lead1m")
    y_test = test.pop("ret_exc_lead1m")

    train = train.fillna(0)
    test = test.fillna(0)
    if use_rf:
        train, test = ranked_random_fourier_features(
            x_train=train,
            x_test=test,
            number_features=number_features,
            seed=seed,
        )

    # XXX MSRR
    # train = train.multiply(y_train, axis="index")
    # F_train = train.reset_index().groupby("eom").sum()
    # F_train.pop("id")
    # assert F_train.shape[0] == rolling_window

    model = LinearRegression(fit_intercept=False, positive=False)
    # model.fit(F_train.values, np.ones(F_train.shape[0]))
    model.fit(train, y_train)
    prediction = test @ model.coef_
    reg_base = TabPFNRegressor(
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 10000
        },  # Needs to be set low so that not OOM on fitting intermediate nodes
    )

    tabpfn_tree_reg = RandomForestTabPFNRegressor(
        tabpfn=reg_base,
        verbose=1,
        max_predict_time=60,  # Will fit for one minute
        fit_nodes=True,  # Wheather or not to fit intermediate nodes
        adaptive_tree=True,  # Whather or not to validate if adding a leaf helps or not
    )

    tabpfn_tree_reg.fit(train, y_train)
    predictions = tabpfn_tree_reg.predict(test)

    linear_returns = prediction.values.reshape(1, -1) @ y_test
    returns = predictions.reshape(1, -1) @ y_test
    portfolio_returns.append(
        {"date": test_date, "tabpfn": returns[0], "linear": linear_returns[0]}
    )

    pd.DataFrame(portfolio_returns).to_csv(output_path, index=False)
    end_time = time.time()
    logging.info(
        f"============>Date:{test_date:%Y%m%d}\tTime:{end_time-start_time:.2f}s"
    )

import pandas as pd
from sklearn.compose import ColumnTransformer

from src.utils.config import load_config
from src.preprocess import (
    load_raw_data,
    clean_data,
    split_features_target,
    build_preprocessor,
)


def test_load_raw_data_shape_and_columns():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])

    # Basic sanity: non-empty, expected number of columns including target
    assert not df.empty
    assert "income" in df.columns
    # adult.data has 15 columns including target before dropping fnlwgt
    assert len(df.columns) == 15


def test_clean_data_drops_fnlwgt_and_standardizes_income():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])

    cleaned = clean_data(df, drop_columns=config["data"]["drop_columns"])

    # fnlwgt should be dropped
    assert "fnlwgt" not in cleaned.columns

    # income labels should be normalized to the two expected classes
    unique_labels = sorted(cleaned["income"].unique().tolist())
    assert unique_labels == ["<=50K", ">50K"]


def test_clean_data_does_not_mutate_original():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])

    df_before = df.copy(deep=True)

    _ = clean_data(df, drop_columns=config["data"]["drop_columns"])

    assert df.equals(df_before)



def test_build_preprocessor_and_transform_shape():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])
    df = clean_data(df, drop_columns=config["data"]["drop_columns"])

    X, y = split_features_target(df, config["project"]["target_column"])

    preprocessor = build_preprocessor(
        categorical_features=config["data"]["categorical_features"],
        numeric_features=config["data"]["numeric_features"],
        scale_numeric=config["preprocessing"]["scale_numeric"],
        missing_strategy_categorical=config["preprocessing"]["missing_strategy_categorical"],
        missing_strategy_numeric=config["preprocessing"]["missing_strategy_numeric"],
        handle_unknown_categories=config["preprocessing"]["handle_unknown_categories"],
    )

    assert isinstance(preprocessor, ColumnTransformer)

    # Fit and transform a small sample; ensure output shape matches number of rows
    X_sample = X.head(100)
    transformed = preprocessor.fit_transform(X_sample)

    assert transformed.shape[0] == X_sample.shape[0]
    # There should be more columns after one-hot encoding categoricals
    assert transformed.shape[1] > len(config["data"]["numeric_features"])

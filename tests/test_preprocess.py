import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocess import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)
from src.utils.config import load_config


def test_load_raw_data_shape_and_columns():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])

    assert not df.empty
    assert "income" in df.columns
    assert len(df.columns) == 15


def test_clean_data_drops_fnlwgt_and_standardizes_income():
    config = load_config()
    df = load_raw_data(config["paths"]["data_raw"])

    cleaned = clean_data(df, drop_columns=config["data"]["drop_columns"])

    assert "fnlwgt" not in cleaned.columns

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

    X_sample = X.head(100)
    transformed = preprocessor.fit_transform(X_sample)

    assert transformed.shape[0] == X_sample.shape[0]
    assert transformed.shape[1] > len(config["data"]["numeric_features"])

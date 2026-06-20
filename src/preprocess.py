from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    return df


def clean_data(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    cleaned = df.copy()

    if drop_columns:
        cleaned = cleaned.drop(columns=drop_columns, errors="ignore")

    cleaned["income"] = cleaned["income"].astype(str).str.strip().str.replace(".", "", regex=False)

    return cleaned


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool = True,
    missing_strategy_categorical: str = "most_frequent",
    missing_strategy_numeric: str = "median",
    handle_unknown_categories: str = "ignore",
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=missing_strategy_categorical)),
            ("encoder", OneHotEncoder(handle_unknown=handle_unknown_categories)),
        ]
    )

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=missing_strategy_numeric)),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )

    return preprocessor

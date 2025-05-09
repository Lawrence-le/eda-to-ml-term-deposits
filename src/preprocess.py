# preprocess.py
import warnings

warnings.filterwarnings("ignore", message=".*BaseEstimator._validate_data.*")

import pandas as pd
import numpy as np
from config import IMPORTANT_FEATURES
from data_loader import dataset_raw

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)


def label_encoding():
    return {"no": 0, "yes": 1, "unknown": -1}


def data_preprocess(dataset):
    important_features = IMPORTANT_FEATURES
    return dataset[important_features]


def data_cleaned(dataset):
    data = data_preprocess(dataset).copy()
    data["Age"] = data["Age"].str.replace(" years", "").astype(int)
    data = data[data["Age"] <= 100]
    data["Occupation"] = data["Occupation"].replace("admin.", "administrator")
    data = data[data["Campaign Calls"] >= 0]
    data["Previous Contact Days"] = data["Previous Contact Days"].replace(999, -1)
    data["No Previous Contact"] = data["Previous Contact Days"].apply(
        lambda x: 1 if x == -1 else 0
    )
    encoding_map = label_encoding()
    data["Subscription Status"] = data["Subscription Status"].map(encoding_map)
    data["Credit Default"] = data["Credit Default"].map(encoding_map)
    return data.dropna()


def get_preprocessed_data(dataset):
    data = data_cleaned(dataset)

    X = data.drop(["Subscription Status"], axis=1)
    y = data["Subscription Status"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X.select_dtypes(include=["number"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    all_features = np.concatenate([numerical_cols, cat_features])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        X_resampled,
        y_resampled,
        all_features,
    )


def test(X_train_processed, X_test_processed, y_train, y_resampled, all_features):
    print("===============Test==============")
    print("X_train_processed shape:", X_train_processed.shape)
    print("X_test_processed shape:", X_test_processed.shape)
    print("y_train distribution:", y_train.value_counts())
    print("y_resampled distribution (after SMOTE):\n", y_resampled.value_counts())
    print("Feature count:", len(all_features))
    print("Feature names preview:", all_features[:5])

    print("All preprocessing tests passed ✅")
    print("===============End of Test==============")


if __name__ == "__main__":
    print("====== This is the Preprocess Script ======")

    # Cleaned data
    cleaned = data_cleaned(dataset_raw)
    print("Test - Cleaned Data Shape:", cleaned.shape)

    # Get processed outputs
    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        X_resampled,
        y_resampled,
        all_features,
    ) = get_preprocessed_data(dataset_raw)

    test(X_train_processed, X_test_processed, y_train, y_resampled, all_features)

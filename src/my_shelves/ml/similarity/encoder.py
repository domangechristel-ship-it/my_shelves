from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.compose import make_column_selector, ColumnTransformer


# fit transformers on a subset of the dataset

# transform the input on demand

def get_transformer():
    num_col = make_column_selector(dtype_include=['float64'])
    cat_col = make_column_selector(dtype_include=['object','bool'])

    scaler_pipe = Pipeline([
        ("robust_scaler", RobustScaler().set_output(transform="pandas")),
        ]
    )

    feature_transformer = ColumnTransformer(
        [
        ("numerical_encoder", scaler_pipe, num_col),
        ],
        remainder="drop"
    )
    pipeline = Pipeline([
        ("feature_transformer", feature_transformer)
    ]).set_output(transform="pandas")
    return pipeline

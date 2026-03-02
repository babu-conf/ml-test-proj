import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry
from snowflake.ml.model import model_signature

from utils import prepare_data, FEATURE_COLS, CAT_COLS, NUM_COLS, MODEL_NAME, VERSION_NAME

# Get session (auto-available inside ML Job container)
session = Session.builder.getOrCreate()

# Load data using SQL file
with open("sql/load_data.sql", "r") as f:
    query = f.read()

df = session.sql(query).to_pandas()
df = prepare_data(df)

# Train/test split
X = df[FEATURE_COLS]
y = df["PRICE_CATEGORY"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
])
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Register model
input_features = [
    model_signature.FeatureSpec(name="CARAT", dtype=model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec(name="CUT", dtype=model_signature.DataType.STRING),
    model_signature.FeatureSpec(name="COLOR", dtype=model_signature.DataType.STRING),
    model_signature.FeatureSpec(name="CLARITY", dtype=model_signature.DataType.STRING),
    model_signature.FeatureSpec(name="DEPTH", dtype=model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec(name="TABLE_PCT", dtype=model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec(name="X", dtype=model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec(name="Y", dtype=model_signature.DataType.DOUBLE),
    model_signature.FeatureSpec(name="Z", dtype=model_signature.DataType.DOUBLE),
]
output_features = [
    model_signature.FeatureSpec(name="output_feature_0", dtype=model_signature.DataType.INT64),
]
sig = model_signature.ModelSignature(inputs=input_features, outputs=output_features)

reg = Registry(session=session, database_name="ML_DEMO", schema_name="WORKSPACE")
mv = reg.log_model(
    model=pipeline,
    model_name=MODEL_NAME,
    version_name=VERSION_NAME,
    signatures={"predict": sig},
    target_platforms=["WAREHOUSE"],
    options={"relax_version": True},
)
print(f"Model registered: {mv.model_name} v{mv.version_name}")

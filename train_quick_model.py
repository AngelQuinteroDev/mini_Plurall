import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Cargar SOLO application_train
df = pd.read_csv("data/raw/home-credit-default-risk/application_train.csv")

# 2. Seleccionar pocas columnas numéricas
features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED"
]

df = df[features + ["TARGET"]].dropna(subset=["TARGET"])

X = df[features]
y = df["TARGET"]

# 3. Pipeline simple (imputer + modelo)
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ))
])

# 4. Entrenar con POQUITOS datos (rápido)
X_train, _, y_train, _ = train_test_split(
    X, y, train_size=0.1, random_state=42
)

pipeline.fit(X_train, y_train)

# 5. Guardar modelo
joblib.dump(pipeline, "models/credit_model.pkl")

print("✅ Modelo entrenado y guardado")

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ── Cargar datos ───────────────────────────────────────────────────────────────
data = pd.read_csv("dataset_ciclismo_fatiga.csv")

X = data[["frecuencia_cardiaca", "potencia", "cadencia",
          "temperatura", "tiempo", "pendiente", "velocidad"]]
y = data["fatiga"]

# ── Dividir datos ──────────────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Buscar el mejor K ──────────────────────────────────────────────────────────
print("=== KNN (buscando mejor k) ===")

mejor_k   = None
mejor_mse = float("inf")

for k in [3, 5, 7, 9, 11, 15]:
    pipeline_temp = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsRegressor(n_neighbors=k))
    ])
    pipeline_temp.fit(x_train, y_train)

    mse_temp = mean_squared_error(y_test, pipeline_temp.predict(x_test))
    r2_temp  = r2_score(y_test, pipeline_temp.predict(x_test))

    print(f"k={k} -> MSE: {mse_temp:.2f} | R²: {r2_temp:.4f}")

    if mse_temp < mejor_mse:
        mejor_mse = mse_temp
        mejor_k   = k

print(f"\n Mejor k encontrado: {mejor_k}")

# ── Pipeline: KNN con mejor k ──────────────────────────────────────────────────
pipeline_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", KNeighborsRegressor(n_neighbors=mejor_k))
])
pipeline_knn.fit(x_train, y_train)

# ── Pipeline: Regresión Lineal ─────────────────────────────────────────────────
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LinearRegression())
])
pipeline_lr.fit(x_train, y_train)

# ── Evaluación ─────────────────────────────────────────────────────────────────
mse_knn = mean_squared_error(y_test, pipeline_knn.predict(x_test))
r2_knn  = r2_score(y_test, pipeline_knn.predict(x_test))

mse_lr  = mean_squared_error(y_test, pipeline_lr.predict(x_test))
r2_lr   = r2_score(y_test, pipeline_lr.predict(x_test))

print("\n Resultados Finales")
print(f"KNN (k={mejor_k})")
print("MSE:", mse_knn)
print("R²:",  r2_knn)

print("\nRegresión Lineal")
print("MSE:", mse_lr)
print("R²:",  r2_lr)

# ── Guardar modelos ────────────────────────────────────────────────────────────
joblib.dump({"knn": pipeline_knn, "lr": pipeline_lr, "mejor_k": mejor_k},
            "modelos_ciclismo.pkl")

print("\n Modelos guardados en 'modelos_ciclismo.pkl'")
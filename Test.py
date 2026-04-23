import streamlit as st
import pandas as pd
import joblib
import subprocess
import os

st.set_page_config(page_title="Predicción de Fatiga Ciclista", page_icon="🚴", layout="wide")

st.title("Predicción de fatiga ciclista")
st.caption("Estimación del nivel de agotamiento físico mediante Machine Learning")
st.divider()

col_info, col_form = st.columns([1, 1.2], gap="large")

# ── Columna izquierda: información ────────────────────────────────────────────
with col_info:

    st.subheader("Descripción")
    st.write("Ingresa las métricas del ciclista para estimar su nivel de fatiga usando dos modelos entrenados.")
    st.markdown("**KNN** — Compara con ciclistas de condiciones similares en los datos de entrenamiento.")
    st.markdown("**Regresión lineal** — Calcula la fatiga por relación matemática entre las variables ingresadas.")

    st.divider()

    st.subheader("Niveles de fatiga (0 – 100)")
    niveles = [
        ("🟢", "0 – 20",   "Muy baja", "Sin fatiga significativa"),
        ("🟡", "21 – 40",  "Baja",     "Esfuerzo leve"),
        ("🟠", "41 – 60",  "Media",    "Fatiga moderada"),
        ("🔴", "61 – 80",  "Alta",     "Fatiga evidente"),
        ("⛔", "81 – 100", "Muy alta", "Fatiga extrema / agotamiento"),
    ]
    for icono, rango, nivel, desc in niveles:
        st.markdown(f"{icono} **{rango}** · {nivel} — {desc}")

    st.divider()

    # ── Botón entrenar ─────────────────────────────────────────────────────────
    st.subheader("Entrenamiento del modelo")
    st.write("Si aún no has entrenado el modelo o quieres actualizarlo con nuevos datos, haz clic aquí.")

    entrenar = st.button("Entrenar modelo", use_container_width=True)

    if entrenar:
        with st.spinner("Entrenando modelo..."):
            ruta_train = os.path.join(os.path.dirname(__file__), "Train.py")

            resultado = subprocess.run(
                ["python", ruta_train],
                capture_output=True,
                text=True
            )

            if resultado.returncode == 0:
                st.success("Modelo entrenado y guardado correctamente.")
                st.code(resultado.stdout)
            else:
                st.error("Error durante el entrenamiento.")
                st.code(resultado.stderr)

# ── Columna derecha: formulario y predicción ───────────────────────────────────
with col_form:

    st.subheader("Datos del ciclista")

    c1, c2 = st.columns(2)
    with c1:
        frecuencia  = st.number_input("Frecuencia cardiaca (bpm)", min_value=40,  max_value=220, value=140)
        cadencia    = st.number_input("Cadencia (rpm)",             min_value=20,  max_value=150, value=80)
        tiempo      = st.number_input("Tiempo (min)",               min_value=1,   max_value=300, value=60)
    with c2:
        potencia    = st.number_input("Potencia (W)",               min_value=50,  max_value=600, value=250)
        temperatura = st.number_input("Temperatura (°C)",           min_value=-10, max_value=50,  value=28)
        pendiente   = st.number_input("Pendiente (%)",              min_value=-20, max_value=30,  value=3)

    velocidad = st.number_input("Velocidad (km/h)", min_value=1, max_value=100, value=25)

    predecir = st.button("Predecir nivel de fatiga", use_container_width=True)

    if predecir:
        try:
            ruta_pkl = os.path.join(os.path.dirname(__file__), "modelos_ciclismo.pkl")

            datos      = joblib.load(ruta_pkl)
            modelo_knn = datos["knn"]
            modelo_lr  = datos["lr"]
            scaler     = datos["scaler"]
            mejor_k    = datos["mejor_k"]

            nuevo = pd.DataFrame([[frecuencia, potencia, cadencia,
                                   temperatura, tiempo, pendiente, velocidad]],
                                 columns=["frecuencia_cardiaca", "potencia", "cadencia",
                                          "temperatura", "tiempo", "pendiente", "velocidad"])

            nuevo_scaled = scaler.transform(nuevo)
            pred_knn = modelo_knn.predict(nuevo_scaled)[0]
            pred_lr  = modelo_lr.predict(nuevo_scaled)[0]

            def interpretar(valor):
                valor = max(0, min(100, valor))
                if valor <= 20:   return "Muy baja",  "Sin fatiga significativa"
                elif valor <= 40: return "Baja",      "Esfuerzo leve"
                elif valor <= 60: return "Media",     "Fatiga moderada"
                elif valor <= 80: return "Alta",      "Fatiga evidente"
                else:             return "Muy alta",  "Fatiga extrema / agotamiento"

            st.divider()
            st.subheader("Resultados")

            r1, r2 = st.columns(2)
            for col, nombre, pred in [(r1, f"KNN (k={mejor_k})", pred_knn), (r2, "Regresión lineal", pred_lr)]:
                nivel, estado = interpretar(pred)
                with col:
                    st.metric(label=nombre, value=f"{pred:.1f} / 100")
                    st.write(f"**{nivel}** — {estado}")

        except FileNotFoundError:
            st.error("Primero debes entrenar el modelo usando el botón de la izquierda.")
        except Exception as e:
            st.error(f"Error al predecir: {e}")
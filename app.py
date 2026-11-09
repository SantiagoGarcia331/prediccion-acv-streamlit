"""Aplicación académica de despliegue del clasificador de ACV (tercer corte).

Ejecutar: streamlit run app.py
El entrenamiento se realiza únicamente con train_model.py, fuera de la app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from validacion import predecir

BASE = Path(__file__).resolve().parent
DATOS = BASE / "data" / "healthcare-dataset-stroke-data.csv"
PIPELINE = BASE / "models" / "pipeline_acv.joblib"
METRICAS = BASE / "models" / "metricas.json"
CASO = BASE / "models" / "caso_prueba.json"
COLORES = {"azul": "#16324B", "verde": "#138A86", "coral": "#DC695D", "gris": "#637484"}

st.set_page_config(
    page_title="ACV Insight | Minería de Datos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1190px; padding-top: 2.0rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -.02em; color: #173549;}
      .stApp {background: #fcfdfd;}
      [data-testid="stSidebar"] {background-color: #eff6f6;}
      .hero {border-radius: 18px; padding: 28px 34px; margin-bottom: 20px;
             background: linear-gradient(115deg,#16324B 0%,#1E5968 100%); color: white;}
      .hero h1 {color: white; font-size: 2.2rem; margin: 8px 0 10px; line-height:1.12;}
      .hero p {color: #e3eff1; max-width: 780px; margin-bottom: 0; font-size: 1.0rem;}
      .eyebrow {font-size:.77rem; font-weight:700; letter-spacing:.13em; color:#8bd5d1;}
      .panel {background: #f0f6f7; border-left: 3px solid #138A86;
              padding: 17px 20px; border-radius: 8px; margin: 10px 0 20px;}
      .panel strong {color: #16324B;}
      .micro {color: #647785; font-size: .83rem;}
      div[data-testid="stMetric"] {background: white; border:1px solid #dfebed;
                                   border-radius: 12px; padding: 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Cargando el pipeline entrenado...")
def cargar_modelo():
    return joblib.load(PIPELINE)

@st.cache_data(show_spinner=False)
def cargar_datos():
    return pd.read_csv(DATOS)

@st.cache_data(show_spinner=False)
def cargar_metadatos():
    return json.loads(METRICAS.read_text(encoding="utf-8")), json.loads(CASO.read_text(encoding="utf-8"))

try:
    modelo = cargar_modelo()
    datos = cargar_datos()
    metricas, caso = cargar_metadatos()
except (OSError, ValueError, EOFError, ImportError) as error:
    st.error("No se pudieron cargar los archivos del proyecto. Revisa que el repositorio incluya datos/, models/ y las dependencias de requirements.txt.")
    st.exception(error)
    st.stop()

resultado = metricas["modelos"][metricas["seleccionado"]]

with st.sidebar:
    st.markdown("## ACV Insight")
    st.caption("Proyecto académico · Minería de Datos · Corte 3")
    st.divider()
    seccion = st.radio(
        "Explorar aplicación",
        ["Inicio", "Exploración de datos", "Simulador de predicción", "Evaluación del modelo", "Sobre el proyecto"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("**Modelo desplegado:** Regresión logística")
    st.caption("**Base de datos:** Stroke Prediction Dataset (Kaggle)")
    st.warning("Prototipo exclusivamente académico. No sirve para diagnosticar, descartar ni decidir tratamientos de ACV.")

if seccion == "Inicio":
    st.markdown(
        """<section class="hero">
          <div class="eyebrow">MINERÍA DE DATOS · TERCER CORTE</div>
          <h1>Explorando patrones asociados al ACV</h1>
          <p>Una experiencia interactiva para estudiar un modelo de clasificación de accidente cerebrovascular
          a partir de variables demográficas, clínicas y de estilo de vida.</p>
        </section>""",
        unsafe_allow_html=True,
    )
    st.write("El proyecto integra el análisis de los dos primeros cortes y despliega el pipeline entrenado en una aplicación web. Las predicciones son demostraciones del comportamiento estadístico del modelo, no valoraciones de salud individuales.")
    a, b, c, d = st.columns(4)
    a.metric("Registros originales", f'{metricas["dataset_original"]:,}'.replace(",", "."))
    b.metric("Casos con ACV", str(metricas["casos_stroke_originales"]))
    c.metric("Accuracy (prueba)", f'{resultado["accuracy"]:.1%}')
    d.metric("Recall de ACV", f'{resultado["recall_stroke"]:.1%}')
    st.markdown("### ¿Qué encontrarás?")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("**01 · Comprender**")
        st.write("Visualizaciones de las clases, los grupos de edad y los indicadores del conjunto de datos.")
    with col2:
        st.markdown("**02 · Experimentar**")
        st.write("Formulario con diez variables, validación de entradas y predicciones del pipeline.")
    with col3:
        st.markdown("**03 · Evaluar**")
        st.write("Métricas, validación cruzada, matriz de confusión y limitaciones documentadas.")
    st.markdown(
        "<div class='panel'><strong>¿Por qué no basta el accuracy?</strong> "
        "Solo cerca del 4,9 % de los registros originales tienen la etiqueta ACV. "
        "Un clasificador que siempre responde «No» logra una exactitud alta, pero no identifica ningún caso positivo.</div>",
        unsafe_allow_html=True,
    )

elif seccion == "Exploración de datos":
    st.title("Exploración del conjunto de datos")
    st.write("Estas gráficas describen la distribución de las observaciones. No establecen causalidad ni representan estimaciones clínicas de una población.")
    st.caption("Datos: Stroke Prediction Dataset (Fedesoriano, Kaggle). 5.110 registros originales; se excluye el único registro de género ‘Other’ durante el modelado, siguiendo el corte 2.")
    datos_vis = datos.loc[datos["gender"] != "Other"].copy()
    datos_vis["Resultado"] = datos_vis["stroke"].map({0: "Sin registro de ACV", 1: "Con registro de ACV"})

    conteos = datos_vis["Resultado"].value_counts().reindex(["Sin registro de ACV", "Con registro de ACV"])
    graf1 = px.bar(
        x=conteos.index, y=conteos.values,
        labels={"x": "Etiqueta en el dataset", "y": "Número de registros"},
        title="1. Distribución de clases",
        color=conteos.index,
        color_discrete_map={"Sin registro de ACV": COLORES["verde"], "Con registro de ACV": COLORES["coral"]},
        text=conteos.values,
    )
    graf1.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=60, b=10))
    graf1.update_traces(textposition="outside")
    graf2_df = datos_vis.copy()
    graf2_df["Grupo de edad"] = pd.cut(
        graf2_df["age"], bins=[0, 18, 40, 60, 83],
        labels=["Hasta 18", "19 a 40", "41 a 60", "61 a 82"],
        include_lowest=True,
    )
    tabla_edad = graf2_df.groupby("Grupo de edad", observed=True)["stroke"].agg(["mean", "count"]).reset_index()
    tabla_edad["Porcentaje de registros con ACV"] = tabla_edad["mean"] * 100
    graf2 = px.bar(
        tabla_edad, x="Grupo de edad", y="Porcentaje de registros con ACV",
        text="Porcentaje de registros con ACV", title="2. Proporción observada por edad",
        color_discrete_sequence=[COLORES["azul"]],
        hover_data={"count": True},
    )
    graf2.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    graf2.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10), yaxis_ticksuffix="%")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(graf1, use_container_width=True)
    with col2:
        st.plotly_chart(graf2, use_container_width=True)

    graf3 = px.box(
        datos_vis, x="Resultado", y="avg_glucose_level", color="Resultado",
        color_discrete_map={"Sin registro de ACV": COLORES["verde"], "Con registro de ACV": COLORES["coral"]},
        title="3. Glucosa promedio según la etiqueta ACV",
        labels={"avg_glucose_level": "Glucosa promedio (mg/dL)", "Resultado": "Resultado registrado"},
        points=False,
    )
    graf3.update_layout(showlegend=False, height=390, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(graf3, use_container_width=True)
    st.info("Lectura de los gráficos: el conjunto está muy desbalanceado y los grupos de edad tienen tamaños diferentes. Las asociaciones descriptivas no demuestran que alguna variable cause un ACV.")

elif seccion == "Simulador de predicción":
    st.title("Simulador de predicción")
    st.write("Ingresa valores dentro de los rangos observados en los datos. El sistema valida las entradas y consulta el **pipeline ya entrenado**, sin volver a entrenarlo.")
    with st.expander("¿Qué significa el resultado?", expanded=False):
        st.write("El clasificador asigna una de dos etiquetas del conjunto de datos: 0 (sin registro de ACV) o 1 (con registro de ACV). La probabilidad mostrada es una salida estadística del modelo, no una probabilidad clínica calibrada ni un diagnóstico personal.")
    usar_caso = st.checkbox("Cargar ejemplo de un registro reservado del conjunto de prueba", value=False,
                            help="Permite comprobar que la app produce la misma predicción que el notebook.")
    ejemplo = caso["entrada"] if usar_caso else {
        "gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 105.0, "bmi": 27.0, "smoking_status": "never smoked",
    }
    # Las claves de los componentes cambian al activar el ejemplo para restablecer los valores.
    sufijo = "prueba" if usar_caso else "normal"
    with st.form("formulario_acv"):
        st.markdown("#### 1 · Información general")
        izquierda, derecha = st.columns(2)
        with izquierda:
            genero = st.selectbox("Género registrado", ["Female", "Male"],
                                  index=["Female", "Male"].index(ejemplo["gender"]),
                                  format_func={"Female": "Femenino", "Male": "Masculino"}.get,
                                  key="genero_"+sufijo)
            edad = st.number_input("Edad (años)", min_value=1, max_value=82.0,
                                   value=float(ejemplo["age"]), step=1,
                                   format="%d", key="edad_"+sufijo)
            estado = st.selectbox("¿Alguna vez estuvo casado/a?", ["Yes", "No"],
                                  index=["Yes", "No"].index(ejemplo["ever_married"]),
                                  format_func={"Yes": "Sí", "No": "No"}.get,
                                  key="estado_"+sufijo)
        with derecha:
            tipos = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
            traduccion_tipos = {"Private": "Empleo privado", "Self-employed": "Trabajo independiente",
                                "Govt_job": "Empleo público", "children": "Menor de edad",
                                "Never_worked": "Nunca ha trabajado"}
            trabajo = st.selectbox("Tipo de actividad", tipos, index=tipos.index(ejemplo["work_type"]),
                                   format_func=traduccion_tipos.get, key="trabajo_"+sufijo)
            residencia = st.selectbox("Zona de residencia", ["Urban", "Rural"],
                                      index=["Urban", "Rural"].index(ejemplo["Residence_type"]),
                                      format_func={"Urban": "Urbana", "Rural": "Rural"}.get,
                                      key="residencia_"+sufijo)
            tabaco = ["never smoked", "formerly smoked", "smokes", "Unknown"]
            etiquetas_tabaco = {"never smoked": "Nunca ha fumado", "formerly smoked": "Fumaba anteriormente",
                                "smokes": "Fuma", "Unknown": "Sin información"}
            fumar = st.selectbox("Antecedente de tabaquismo", tabaco,
                                 index=tabaco.index(ejemplo["smoking_status"]),
                                 format_func=etiquetas_tabaco.get, key="fumar_"+sufijo)
        st.markdown("#### 2 · Indicadores de salud registrados")
        izq2, der2 = st.columns(2)
        with izq2:
            hipertension = st.selectbox("Hipertensión informada", [0, 1],
                                        index=int(ejemplo["hypertension"]),
                                        format_func=lambda x: "Sí" if x else "No", key="hip_"+sufijo)
            cardiaca = st.selectbox("Antecedente de enfermedad cardíaca", [0, 1],
                                    index=int(ejemplo["heart_disease"]),
                                    format_func=lambda x: "Sí" if x else "No", key="corazon_"+sufijo)
        with der2:
            glucosa = st.number_input("Glucosa promedio (mg/dL)", min_value=55.12, max_value=271.74,
                                      value=float(ejemplo["avg_glucose_level"]), step=1.0,
                                      key="glucosa_"+sufijo)
            sin_imc = st.checkbox("IMC desconocido (imputación automática)", value=ejemplo["bmi"] is None,
                                  key="imcfalta_"+sufijo)
            imc = None if sin_imc else st.number_input(
                "Índice de masa corporal (kg/m²)", min_value=10.3, max_value=97.6,
                value=float(ejemplo["bmi"] if ejemplo["bmi"] is not None else 26.0), step=.5,
                key="imc_"+sufijo)
        st.caption("Los límites numéricos corresponden a los valores observados en el dataset. No son límites de normalidad médica.")
        enviar = st.form_submit_button("Calcular predicción del modelo", type="primary", use_container_width=True)

    if enviar:
        registro = {
            "gender": genero, "age": edad, "hypertension": hipertension,
            "heart_disease": cardiaca, "ever_married": estado, "work_type": trabajo,
            "Residence_type": residencia, "avg_glucose_level": glucosa,
            "bmi": imc, "smoking_status": fumar,
        }
        try:
            prediccion, probabilidad = predecir(modelo, registro)
        except (ValueError, TypeError) as error:
            st.error(f"Corrige los datos ingresados: {error}")
        else:
            st.divider()
            st.subheader("Resultado del modelo")
            p1, p2 = st.columns([1.4, 1])
            with p1:
                if prediccion == 1:
                    st.warning("**Clase predicha: 1 — Con registro de ACV**")
                else:
                    st.success("**Clase predicha: 0 — Sin registro de ACV**")
                st.caption("La etiqueta predicha describe el patrón aprendido a partir del dataset, no el estado real de salud de una persona.")
            with p2:
                st.metric("Salida del modelo para clase 1", f"{probabilidad:.1%}")
                st.progress(float(probabilidad))
                st.caption("Puntuación no calibrada como riesgo médico individual.")
            if usar_caso:
                coincide = (prediccion == caso["prediccion_notebook"]
                            and abs(probabilidad - caso["probabilidad_notebook"]) < 1e-9)
                if coincide:
                    st.info("**Comprobación reproducible:** para este registro reservado, el resultado coincide con el notebook del segundo corte.")
                else:
                    st.error("El ejemplo no coincide con la referencia: revisa versiones o archivos del pipeline.")
            st.warning("**Importante:** si alguien presenta síntomas compatibles con ACV debe buscar atención médica urgente. Esta aplicación no ayuda a tomar decisiones clínicas.")

elif seccion == "Evaluación del modelo":
    st.title("Evaluación y limitaciones")
    st.write("Comparación sobre el **conjunto de prueba reservado de 1.022 registros**, separado de forma estratificada (80 % entrenamiento / 20 % prueba, semilla 42).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f'{resultado["accuracy"]:.1%}')
    m2.metric("Precisión (ACV)", f'{resultado["precision_stroke"]:.1%}')
    m3.metric("Recall (ACV)", f'{resultado["recall_stroke"]:.1%}')
    m4.metric("F1-score (ACV)", f'{resultado["f1_stroke"]:.1%}')
    st.markdown("### Comparación de modelos")
    filas = []
    for nombre, r in metricas["modelos"].items():
        filas.append({"Modelo": nombre, "Accuracy": r["accuracy"], "Precisión (clase 1)": r["precision_stroke"],
                      "Recall (clase 1)": r["recall_stroke"], "F1 (clase 1)": r["f1_stroke"],
                      "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"]})
    tabla = pd.DataFrame(filas).set_index("Modelo")
    st.dataframe(tabla.style.format("{:.1%}"), use_container_width=True)
    st.caption("Seleccionado: regresión logística por su mejor F1, PR-AUC y ROC-AUC, además del desempeño en validación cruzada. El árbol de decisión se muestra como comparación.")
    st.markdown("### Matriz de confusión · regresión logística")
    cm = resultado["matriz_confusion"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicho: sin ACV", "Predicho: con ACV"],
        y=["Real: sin ACV", "Real: con ACV"],
        text=cm, texttemplate="%{text}", textfont={"size": 19},
        colorscale=[[0, "#edf5f6"], [1, "#117b83"]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Registros: %{z}<extra></extra>",
    ))
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_side="bottom", yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Lectura de la matriz:** 714 verdaderos negativos, 258 falsos positivos, 10 falsos negativos y 40 verdaderos positivos. Se identificaron 40 de 50 casos positivos de prueba, pero hubo 258 alertas positivas en registros sin ACV.")
    st.markdown("### Validación cruzada (5 particiones estratificadas)")
    col1, col2, col3 = st.columns(3)
    col1.metric("F1 medio", f'{resultado["cv_f1_media"]:.1%}')
    col2.metric("Recall medio", f'{resultado["cv_recall_media"]:.1%}')
    col3.metric("PR-AUC medio", f'{resultado["cv_pr_auc_media"]:.1%}')
    st.caption(f'Desviación estándar F1 entre particiones: {resultado["cv_f1_desviacion"]:.3f}. La validación cruzada se hizo exclusivamente sobre el conjunto de entrenamiento.')
    st.markdown("### Interpretación responsable")
    st.warning("Aunque el accuracy supera el 70 %, la precisión positiva (13,4 %) y el F1 (23,0 %) son bajos. El modelo genera muchos falsos positivos y no está validado en pacientes reales, de forma externa ni clínicamente. La puntuación mostrada no es una probabilidad calibrada de padecer ACV.")
    st.write("La clase mayoritaria, que siempre responde ‘sin ACV’, alcanza 95,1 % de accuracy y 0 % de recall positivo. Por eso el accuracy aislado resulta engañoso en un conjunto donde solo alrededor del 4,9 % de los ejemplos pertenecen a la clase positiva.")

else:  # Sobre el proyecto
    st.title("Sobre el proyecto")
    st.markdown("**Nombre:** Predicción de accidente cerebrovascular mediante técnicas de minería de datos basadas en variables clínicas, demográficas y de estilo de vida.")
    st.markdown("**Asignatura:** Minería de Datos · Universidad de la Costa (CUC) · Grupo 25786")
    st.markdown("**Docente:** José Escorcia Gutiérrez")
    st.markdown("**Integrantes:** Adrián García De la Rosa · Camilo Cornado Barraza · Liann Rivero Torres · Santiago García Jiménez")
    st.markdown("**Fuente:** [Stroke Prediction Dataset, Fedesoriano (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)")
    st.markdown("""**Método:** limpieza de registros, separación estratificada entrenamiento/prueba, imputación aprendida en entrenamiento, escalado de numéricas, codificación one-hot de categóricas y regresión logística ponderada por clases. El modelo completo se guarda en `models/pipeline_acv.joblib` y se carga para la predicción.

**Continuidad del trabajo:** el análisis exploratorio se inició en el primer corte; en el segundo se compararon clasificadores y se documentaron sus métricas; en el tercero se construyó y probó esta interfaz reproducible.

**Limitaciones:** datos históricos secundarios; desbalance pronunciado de clases; precisión positiva baja; numerosos falsos positivos; ausencia de calibración probabilística, evaluación externa o validación clínica. Las categorías y rangos se restringen a las opciones y valores observados al entrenar.""")
    st.info("Uso académico exclusivamente. No introducir datos personales reales ni utilizar la herramienta para diagnóstico o seguimiento clínico.")

st.divider()
st.caption("ACV Insight · Universidad de la Costa · Proyecto de aula 2026-2 | Prototipo no clínico")

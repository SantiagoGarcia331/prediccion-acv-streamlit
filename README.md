# 🧠 ACV Insight — Despliegue del modelo en Streamlit

**Minería de Datos · Universidad de la Costa · Tercer corte · 2026-2**

Proyecto académico de clasificación de registros asociados con **accidente cerebrovascular (ACV)**. Continúa el trabajo de los cortes 1 y 2 y convierte el pipeline del segundo corte en una aplicación web interactiva. **No sirve para diagnóstico, clasificación de pacientes ni decisiones médicas.**

## Equipo

- Adrián García De la Rosa
- Camilo Cornado Barraza
- Liann Rivero Torres
- Santiago García Jiménez

**Docente:** José Escorcia Gutiérrez · **Grupo:** 25786

## Dataset y modelo

Fuente: [Stroke Prediction Dataset — Fedesoriano (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

- Conjunto original: **5.110 registros**, de los cuales 249 tienen la etiqueta `stroke=1` (**4,87 %**).
- Modelado: **5.109 registros** (se excluye el único `gender=Other`, decisión documentada en el segundo corte); identificador `id` excluido.
- Variables predictoras: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`.
- División: **80/20**, estratificada, `random_state=42` (4.087 entrenamiento, 1.022 prueba).
- Preprocesamiento: imputación mediana/moda, estandarización de numéricas y codificación one-hot dentro del `Pipeline`, ajustado solo sobre entrenamiento.
- Candidatos: regresión logística (`class_weight='balanced'`, `max_iter=2000`) y árbol (`max_depth=5`, `min_samples_leaf=20`, `class_weight='balanced'`).
- Despliegue: **regresión logística**, seleccionada por mejor F1, PR-AUC y ROC-AUC de prueba y resultados de validación cruzada.

### Resultados medidos sobre los 1.022 registros de prueba

| Modelo | Accuracy | Precisión clase 1 | Recall clase 1 | F1 clase 1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | 0,738 | 0,134 | 0,800 | 0,230 | 0,839 | 0,259 |
| Árbol de decisión | 0,713 | 0,124 | 0,800 | 0,214 | 0,806 | 0,173 |

Matriz de confusión (regresión logística): `[[714, 258], [10, 40]]`; filas = reales `0/1`, columnas = predicciones `0/1`. Validación cruzada estratificada 5-fold (solo entrenamiento): F1 medio **0,237**; recall medio **0,819**; PR-AUC medio **0,192**.

**Limitaciones:** solo ~4,9 % de positivos en el conjunto original; 258 falsos positivos y 10 falsos negativos en prueba; baja precisión positiva (13,4 %). La probabilidad emitida por `predict_proba()` proviene de un clasificador con ponderación de clases, sin calibración para representar riesgo clínico individual. No se ha realizado validación clínica ni externa. Aunque el accuracy supera el 70 %, no demuestra utilidad clínica.

## Estructura del repositorio

```text
app.py                                # Interfaz Streamlit: cinco secciones
validacion.py                         # Rangos, tipos, categorías y predicción
train_model.py                        # Entrenamiento reproducible; nunca se llama desde la app
requirements.txt                      # Dependencias fijadas/limitadas
.streamlit/config.toml                # Tema visual
models/pipeline_acv.joblib            # Pipeline completo (preprocesamiento + clasificador)
models/metricas.json                  # Resultados de evaluación
models/caso_prueba.json               # Registro reservado con salida de referencia
notebooks/Notebook_original_corte_2.ipynb
                                   # Notebook recibido del segundo corte (sin modificar)
data/healthcare-dataset-stroke-data.csv
                                   # Dataset de origen
tests/test_pipeline.py                # Pruebas automáticas
```

## Ejecutar en computador

Con **Python 3.13** y en la carpeta que contiene `app.py`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una URL local como `http://localhost:8501`. El navegador abrirá la aplicación. No ejecutes `train_model.py` para usarla: el pipeline ya está serializado.

Para **reproducir el entrenamiento** de manera opcional:

```bash
python train_model.py
```

Este comando vuelve a crear los archivos dentro de `models/` con las mismas decisiones de entrenamiento del segundo corte. No se cambiaron hiperparámetros con respecto al notebook original; la exportación a `joblib` es la adaptación requerida para el despliegue. Si se cambia el entrenamiento, debe documentarse y comparar sus métricas con la versión anterior.

## Pruebas de funcionamiento

Ejecutar desde la raíz:

```bash
python -m unittest discover -s tests -v
```

Casos cubiertos:

1. **Entrada normal:** una combinación válida de 10 variables devuelve una clase (`0` o `1`) y una probabilidad entre 0 y 1.
2. **Entrada inválida:** edad negativa y categoría de tabaquismo desconocida generan errores de validación; la interfaz también restringe rangos y categorías.
3. **Registro conocido de prueba:** `models/caso_prueba.json` contiene el primer registro reservado de `X_test`, su etiqueta real, predicción y salida `predict_proba()` de referencia. La prueba comprueba la igualdad con el modelo desplegado. En la interfaz se puede activar **«Cargar ejemplo de un registro reservado...»** y verificar el mismo resultado.
4. **IMC faltante:** se permite `bmi=None`; la mediana aprendida en entrenamiento se usa dentro del pipeline.

## Publicar en GitHub y Streamlit Community Cloud

1. Crear un **repositorio público** nuevo en GitHub (por ejemplo `acv-insight-streamlit`).
2. Subir el **contenido de esta carpeta** a la raíz del repositorio, conservando las carpetas `models/`, `data/`, `.streamlit/`, `notebooks/` y `tests/`. No subir `.venv`, contraseñas, tokens o datos de pacientes.
3. Entrar a [share.streamlit.io](https://share.streamlit.io/), iniciar sesión y elegir crear una nueva aplicación vinculada con GitHub.
4. Seleccionar el repositorio, la rama `main` y como **Main file path**: `app.py`. Configurar Python 3.13 si se solicita y está disponible.
5. Desplegar; abrir la URL pública y revisar las cinco secciones. Probar el formulario y el caso de prueba, **preferentemente en una ventana de incógnito** para comprobar el acceso público.
6. Entregar en Moodle **ambos enlaces** (GitHub + Streamlit) antes del plazo indicado por el docente. El archivo ZIP por sí solo no sustituye esos enlaces.

## Uso de inteligencia artificial (declaración requerida por la asignatura)

- **Herramienta:** ChatGPT (OpenAI), como asistencia para organizar el despliegue, documentar la integración y preparar validaciones automáticas.
- **Propósito:** conservar el pipeline del segundo corte, empaquetarlo, construir una interfaz comprensible y elaborar la documentación del despliegue.
- **Prompt principal (síntesis de la solicitud, no cita textual):** «Usando el notebook y el dataset del segundo corte de Minería de Datos, prepara una aplicación Streamlit que reutilice el pipeline entrenado, valide los datos de entrada, muestre predicciones, métricas, matriz de confusión, 2–3 visualizaciones del EDA, limitaciones del modelo y explique cómo publicarla en GitHub y Streamlit Community Cloud».
- **Verificación:** se comparó la configuración del entrenamiento con el notebook original; se regeneraron las métricas desde los datos y la separación estratificada; se incorporaron pruebas automáticas para entradas válidas e inválidas y una predicción de referencia del conjunto de prueba. El equipo debe ejecutar las pruebas en su propio entorno y verificar la URL pública después del despliegue.
- **Problemas identificados:** clases desbalanceadas, 201 valores faltantes en `bmi`, precisión positiva baja, posibilidad de fuga de información si se imputara antes de dividir y riesgo de interpretar la puntuación como un diagnóstico médico.
- **Ajustes aplicados:** el procesamiento se mantiene **dentro** del pipeline; se utiliza imputación aprendida solo de entrenamiento; se exporta el pipeline completo con `joblib`; se muestran advertencias de uso académico; se validan rangos, tipos y categorías. **No se cambiaron los hiperparámetros ni se afirmó una mejora de las métricas** respecto al segundo corte.

**Responsabilidad:** antes de entregar, todos los integrantes deben revisar el README, comprender el código y confirmar los resultados y enlaces que efectivamente hayan probado.

---

*Aplicación académica. Sin datos personales de pacientes ni pretensiones de uso sanitario.*

"""Validación común entre la aplicación y las pruebas académicas."""
from numbers import Real
from math import isfinite
import pandas as pd

COLUMNAS = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
]

CATEGORIAS = {
    "gender": ["Female", "Male"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
}

# Límites OBSERVADOS en el dataset usado para entrenar; no son límites médicos.
RANGOS = {
    "age": (0.08, 82.00),
    "avg_glucose_level": (55.12, 271.74),
    "bmi": (10.3, 97.6),
}

NOMBRES = {
    "age": "Edad", "avg_glucose_level": "Glucosa promedio", "bmi": "Índice de masa corporal",
    "hypertension": "Hipertensión", "heart_disease": "Enfermedad cardíaca",
    "gender": "Género", "ever_married": "Antecedente matrimonial",
    "work_type": "Tipo de actividad", "Residence_type": "Zona de residencia",
    "smoking_status": "Historial de tabaquismo"
}


def validar_registro(registro: dict) -> pd.DataFrame:
    """Acepta un registro válido y devuelve un DataFrame con las 10 columnas en orden.

    Impide valores fuera del dominio observado, categorías no entrenadas y tipos incompatibles.
    bmi=None es válido: el pipeline imputa el dato usando el entrenamiento.
    """
    if not isinstance(registro, dict):
        raise ValueError("El registro debe ser un diccionario de variables.")
    faltantes = [c for c in COLUMNAS if c not in registro]
    extras = [c for c in registro if c not in COLUMNAS]
    if faltantes or extras:
        raise ValueError(f"Columnas faltantes: {faltantes}. Columnas no esperadas: {extras}.")

    errores = []
    for c, opciones in CATEGORIAS.items():
        if not isinstance(registro[c], str) or registro[c] not in opciones:
            errores.append(f"{NOMBRES[c]}: categoría desconocida ({registro[c]!r}).")
    for c in ("hypertension", "heart_disease"):
        if type(registro[c]) not in (int,) or registro[c] not in (0, 1):
            errores.append(f"{NOMBRES[c]}: use 0 (no) o 1 (sí).")
    for c, (inferior, superior) in RANGOS.items():
        v = registro[c]
        if c == "bmi" and v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v):
            errores.append(f"{NOMBRES[c]}: debe ser un número válido.")
        elif not inferior <= float(v) <= superior:
            errores.append(
                f"{NOMBRES[c]}: el valor debe estar entre {inferior:g} y {superior:g} "
                "(rango observado en los datos)."
            )
    if errores:
        raise ValueError("\n".join(errores))

    return pd.DataFrame([{c: (float('nan') if c == "bmi" and registro[c] is None else registro[c])
                          for c in COLUMNAS}], columns=COLUMNAS)


def predecir(modelo, registro: dict):
    """Ejecuta exactamente el pipeline serializado, sin volverlo a entrenar."""
    fila = validar_registro(registro)
    prediccion = int(modelo.predict(fila)[0])
    probabilidad_clase_1 = float(modelo.predict_proba(fila)[0, 1])
    return prediccion, probabilidad_clase_1

"""Reproduce el segundo corte y exporta el pipeline completo para Streamlit.

Ejecutar SOLO para entrenar/actualizar: python train_model.py
La app.py únicamente carga el archivo generado, no llama a este módulo.
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from validacion import COLUMNAS

ROOT = Path(__file__).resolve().parent
DATOS = ROOT / 'data' / 'healthcare-dataset-stroke-data.csv'
CARPETA_MODELOS = ROOT / 'models'


def cargar_y_separar():
    bruto = pd.read_csv(DATOS)
    # Se mantiene exactamente la decisión del segundo corte (un registro Other).
    limpio = bruto.loc[bruto['gender'] != 'Other'].copy()
    X = limpio.drop(columns=['id', 'stroke'])
    assert list(X.columns) == COLUMNAS, 'El orden de las variables cambió frente al corte 2.'
    y = limpio['stroke'].astype(int)
    return bruto, limpio, train_test_split(X, y, test_size=.20,
                                           stratify=y, random_state=42)


def crear_pipeline(modelo, X):
    numericas = X.select_dtypes(include=np.number).columns.tolist()
    categoricas = X.select_dtypes(exclude=np.number).columns.tolist()
    preprocesamiento = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputacion', SimpleImputer(strategy='median')),
            ('escala', StandardScaler()),
        ]), numericas),
        ('cat', Pipeline([
            ('imputacion', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore')),
        ]), categoricas),
    ])
    return Pipeline([('preprocesamiento', preprocesamiento), ('clasificador', modelo)])


def entrenar():
    bruto, limpio, particiones = cargar_y_separar()
    X_train, X_test, y_train, y_test = particiones
    modelos = {
        'Regresión logística': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'Árbol de decisión': DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                     class_weight='balanced', random_state=42),
    }
    validacion_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = {}
    resultados = {}
    for nombre, estimador in modelos.items():
        pipe = crear_pipeline(estimador, X_train)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        cv = cross_validate(crear_pipeline(estimador, X_train), X_train, y_train,
                            cv=validacion_cv,
                            scoring={'F1':'f1', 'Recall':'recall', 'PR_AUC':'average_precision'},
                            n_jobs=1, error_score='raise')
        resultados[nombre] = {
            'accuracy': float(accuracy_score(y_test, pred)),
            'precision_stroke': float(precision_score(y_test, pred, zero_division=0)),
            'recall_stroke': float(recall_score(y_test, pred, zero_division=0)),
            'f1_stroke': float(f1_score(y_test, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, prob)),
            'pr_auc': float(average_precision_score(y_test, prob)),
            'matriz_confusion': confusion_matrix(y_test, pred).tolist(),
            'cv_f1_media': float(cv['test_F1'].mean()),
            'cv_f1_desviacion': float(cv['test_F1'].std()),
            'cv_recall_media': float(cv['test_Recall'].mean()),
            'cv_pr_auc_media': float(cv['test_PR_AUC'].mean()),
        }
        pipelines[nombre] = pipe

    base = DummyClassifier(strategy='most_frequent', random_state=42)
    base.fit(X_train, y_train)
    baseline_pred = base.predict(X_test)

    metadata = {
        'proyecto': 'Predicción de accidente cerebrovascular (ACV)',
        'seleccionado': 'Regresión logística',
        'motivo_seleccion': 'Mejor F1 positivo, PR-AUC, ROC-AUC y validación cruzada que el árbol.',
        'dataset_original': int(len(bruto)),
        'registros_modelado': int(len(limpio)),
        'casos_stroke_originales': int(bruto['stroke'].sum()),
        'porcentaje_positivos_original': float(bruto['stroke'].mean() * 100),
        'entrenamiento': int(len(X_train)),
        'prueba': int(len(X_test)),
        'version_sklearn': sklearn.__version__,
        'base_mayoritaria': {
            'accuracy': float(accuracy_score(y_test, baseline_pred)),
            'recall_stroke': float(recall_score(y_test, baseline_pred, zero_division=0)),
            'f1_stroke': float(f1_score(y_test, baseline_pred, zero_division=0)),
        },
        'modelos': resultados,
    }

    # Caso de prueba auténtico reservado y reproducible (primer registro de X_test).
    ref = X_test.iloc[0]
    registro = {c: (None if pd.isna(ref[c]) else
                    int(ref[c]) if c in ('hypertension','heart_disease') else
                    float(ref[c]) if c in ('age','avg_glucose_level','bmi') else str(ref[c]))
                for c in COLUMNAS}
    modelo_elegido = pipelines['Regresión logística']
    referencia = {
        'entrada': registro,
        'etiqueta_real': int(y_test.iloc[0]),
        'prediccion_notebook': int(modelo_elegido.predict(X_test.iloc[[0]])[0]),
        'probabilidad_notebook': float(modelo_elegido.predict_proba(X_test.iloc[[0]])[0, 1]),
        'explicacion': 'Ejemplo reservado del conjunto de prueba, no un nuevo dato clínico.',
    }
    return modelo_elegido, metadata, referencia


def exportar():
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo, metricas, referencia = entrenar()
    joblib.dump(modelo, CARPETA_MODELOS / 'pipeline_acv.joblib', compress=3)
    (CARPETA_MODELOS / 'metricas.json').write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding='utf-8')
    (CARPETA_MODELOS / 'caso_prueba.json').write_text(json.dumps(referencia, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Pipeline exportado:', CARPETA_MODELOS / 'pipeline_acv.joblib')
    for nombre, resultados in metricas['modelos'].items():
        print(nombre, 'accuracy=',round(resultados['accuracy'],4),
              'recall=',round(resultados['recall_stroke'],4),
              'F1=',round(resultados['f1_stroke'],4),
              'Matriz=',resultados['matriz_confusion'])
    return modelo, metricas, referencia

if __name__ == '__main__':
    exportar()

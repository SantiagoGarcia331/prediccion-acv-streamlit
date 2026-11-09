"""Pruebas requeridas por la actividad: caso normal, entrada inválida, caso de prueba."""
import json
import unittest
from pathlib import Path

import joblib

from validacion import predecir, validar_registro

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / 'models' / 'pipeline_acv.joblib'
CASO = ROOT / 'models' / 'caso_prueba.json'


class PruebasAplicacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo = joblib.load(PIPE)
        cls.caso = json.loads(CASO.read_text(encoding='utf-8'))

    def test_caso_normal(self):
        normal = {
            'gender':'Female', 'age':45.0, 'hypertension':0, 'heart_disease':0,
            'ever_married':'Yes', 'work_type':'Private', 'Residence_type':'Urban',
            'avg_glucose_level':105.0, 'bmi':27.0, 'smoking_status':'never smoked',
        }
        pred, prob = predecir(self.modelo, normal)
        self.assertIn(pred, (0,1))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_entrada_invalida(self):
        entrada = self.caso['entrada'].copy()
        entrada['age'] = -7
        with self.assertRaisesRegex(ValueError,'Edad'):
            validar_registro(entrada)
        entrada = self.caso['entrada'].copy()
        entrada['smoking_status'] = 'categoria_falsa'
        with self.assertRaisesRegex(ValueError,'tabaquismo'):
            validar_registro(entrada)

    def test_mismo_registro_misma_prediccion_notebook(self):
        pred, prob = predecir(self.modelo, self.caso['entrada'])
        self.assertEqual(pred, self.caso['prediccion_notebook'])
        self.assertAlmostEqual(prob, self.caso['probabilidad_notebook'], places=9)

    def test_imc_faltante_imputado(self):
        entrada = self.caso['entrada'].copy()
        entrada['bmi'] = None
        pred, prob = predecir(self.modelo, entrada)
        self.assertIn(pred, (0,1))
        self.assertGreaterEqual(prob,0)


if __name__ == '__main__':
    unittest.main()

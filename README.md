# Analítica de datos y aprendizaje automático para comprender y planificar la actividad comercial

Repositorio del desarrollo computacional asociado al artículo de grado de Ingeniería de Sistemas de la Universidad Santo Tomás – Seccional Tunja, Colombia.

**Autor:** Julian Felipe Peralta Becerra  
**Tutor del artículo:** Harrizon Soler

## Objetivo

Analizar cómo los registros transaccionales pueden transformarse en indicadores y pronósticos que apoyen la comprensión de las ventas, la anticipación de la recompra y la evaluación de decisiones comerciales bajo distintos supuestos de costos.

## Metodología

1. Preparación y auditoría de los datos: eliminación de duplicados exactos, separación de ventas positivas, movimientos negativos y referencias que no corresponden necesariamente a mercancía.
2. Análisis descriptivo: ventas, concentración del gasto, segmentación RFM, clasificación ABC y asociaciones de productos.
3. Evaluación predictiva: comparación de indicadores RFM calibrados, regresión logística, Random Forest y Gradient Boosting para recompra a 90 días; comparación de métodos para pronóstico semanal de ventas mediante validación temporal.
4. Evaluación de decisiones: simulación de 27 escenarios económicos con márgenes, costos por contacto y aumentos de recompra hipotéticos.

## Estructura

```text
commercial-analytics-machine-learning/
├── app.py
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── .gitignore
├── .gitattributes
├── data/
│   ├── README.md
│   └── online_retail_II.csv
└── results/             # Generado al ejecutar el programa
    ├── tables/          # 28 archivos CSV
    ├── figures/         # 10 archivos PNG
    └── metrics/         # 2 archivos JSON
```

## Datos

Se utiliza Online Retail II (1.067.371 registros), correspondiente a un minorista británico entre diciembre de 2009 y diciembre de 2011. La versión CSV utilizada procede de Kaggle. Consulta [data/README.md](data/README.md) para conocer la fuente y verificar la identidad del archivo. Si el CSV no está incluido, descárgalo y guárdalo como `data/online_retail_II.csv`.

## Instalación

### Entorno computacional de referencia

Los resultados presentados en el artículo fueron reproducidos en Windows utilizando el siguiente entorno computacional:

| Software | Versión |
|---|---|
| Python | 3.13.7 |
| pandas | 2.2.3 |
| NumPy | 2.3.5 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.11.2 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |

Para reproducir los resultados se recomienda utilizar Python 3.13.7 e instalar las versiones exactas de las dependencias especificadas en `requirements.txt`. El uso de otras versiones de las bibliotecas puede generar pequeñas variaciones en las métricas de los modelos predictivos, incluso cuando se conservan los datos, los hiperparámetros y las semillas aleatorias.

Las versiones principales del entorno de ejecución, las semillas, los cortes temporales y el identificador SHA-256 del conjunto de datos se encuentran registrados en [29_entorno_reproducibilidad.json](results/metrics/29_entorno_reproducibilidad.json).

Antes de crear el entorno virtual, comprueba la versión de Python con `python --version` (o `python3 --version` en Linux o macOS).

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

### Linux o macOS

Instrucciones alternativas de instalación; la ejecución de referencia descrita anteriormente se verificó en Windows. La disponibilidad de las dependencias puede variar según la plataforma.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

## Ejecución

Desde la carpeta del repositorio:

```bash
python app.py
```

Opcionalmente, se pueden especificar las rutas de entrada y salida:

```bash
python app.py --input "data/online_retail_II.csv" --output results
```

## Resultados reportados en el artículo

| Indicador | Resultado |
|---|---:|
| Registros originales | 1.067.371 |
| Duplicados exactos | 34.335 |
| Ventas brutas positivas de mercancía | £19.642.692,15 |
| Clientes identificados en ventas de mercancía | 5.852 |
| AUC-ROC de Random Forest (recompra, prueba) | 0,7954 |
| AUC-ROC del indicador RFM calibrado (prueba) | 0,7822 |
| WAPE medio del promedio de cuatro semanas | 16,76 % |
| WAPE medio de Random Forest temporal | 24,56 % |

Los resultados de una ejecución se pueden comprobar en los archivos CSV y JSON generados. Para reproducir las cifras, deben coincidir el archivo de entrada y las versiones del entorno de ejecución. Las métricas de pronóstico de referencia se encuentran en [26_comparacion_pronosticos.csv](results/tables/26_comparacion_pronosticos.csv).

## Alcance y limitaciones

El estudio es retrospectivo y utiliza registros históricos de un solo minorista. Los escenarios económicos se basan en supuestos, no en costos o márgenes observados. No se realizaron intervenciones empresariales ni se demostraron efectos causales, incrementos reales en ventas o reducciones reales de costos. Los resultados no se generalizan automáticamente a otros negocios.

## Citación

La información de citación del software se encuentra en `CITATION.cff`. Una vez publicada y archivada la versión científica, se incorporará el DOI correspondiente.

## Licencias

La licencia del código propio se indica en `LICENSE`, si se ha concedido autorización para su distribución. El conjunto de datos conserva sus propias condiciones de licencia y atribución; consulta `data/README.md`.

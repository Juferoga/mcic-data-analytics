# Analitica - ETL Engine with SOM

**Motor ETL académico enfocado en normalización de datos y Mapas Organizativos (Self-Organizing Maps)**

## 🎯 Visión del Proyecto

Analitica es un motor ETL flexible diseñado para demostrar técnicas de:
- **Normalización universal** de datos heterogéneos
- **Transformación texto→numérico** usando encoders inteligentes
- **Agrupamiento con SOM** para análisis exploratorio de datos
- **Pipeline extensible** para múltiples fuentes de datos

## 📋 Características Principales

### 1. Motor ETL Genérico
```python
from analitica.etl import Pipeline, Source, Destination

pipeline = (
    Pipeline()
    .extract_from(Source.csv("datos.csv"))
    .transform(Normalize(), TextToNumber(), SOMCluster())
    .load_to(Destination.csv("resultado.csv"))
    .run()
)
```

### 2. Normalización de Datos
- **MinMaxScaler**: Escala a rango [0, 1], para datos con límites conocidos
- **ZScoreScaler**: Score estándar (mean=0, std=1), para distribuciones normales
- **RobustScaler**: Mediana + IQR, resistente a outliers
- **LogTransformer**: Transformación logarítmica para datos sesgados
- **PowerTransformer**: Yeo-Johnson para normalizar distribuciones

### 3. Transformadores Texto→Número
- **LabelEncoder**: Categorías ordinales → enteros (low/medium/high → 0/1/2)
- **OneHotEncoder**: Categorías nominales → columnas binarias (city → city_NY, city_LA)
- **TargetEncoder**: Supervisado → media del target por categoría
- **HashEncoder**: Alta cardinalidad → hash a bins fijos (URLs, IDs)

### 4. Integración SOM (Self-Organizing Maps)
- **MiniSom** como librería base
- **U-Matrix** para visualización de clusters
- **Quantization Error** para calidad del mapa
- **BMU Detection** para nuevos datos

## 🏗️ Arquitectura

```
analitica/
├── etl/                    # Motor ETL core
│   ├── pipeline.py        # Pipeline orchestration
│   ├── source.py          # Data extraction
│   └── destination.py     # Data loading
├── normalization/         # Normalización de datos
│   ├── minmax.py
│   ├── zscore.py
│   ├── robust.py
│   └── power.py
├── transformers/          # Transformadores texto→número
│   ├── label_encoder.py
│   ├── onehot_encoder.py
│   ├── target_encoder.py
│   └── hash_encoder.py
├── som/                   # Integración SOM
│   ├── trainer.py
│   ├── analyzer.py
│   └── visualizer.py
├── utils/                 # Utilidades
│   ├── data_quality.py
│   └── validators.py
├── cli.py                 # Interfaz命令行
└── main.py                # Punto de entrada
```

## 🚀 Uso Rápido

### Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows

# Instalar en modo desarrollo
pip install -e ".[dev]"

# Instalar solo dependencias core
pip install -e .
```

### Ejemplo Básico

```python
from analitica.etl import Pipeline
from analitica.normalization import MinMaxScaler, ZScoreScaler, RobustScaler
from analitica.transformers import LabelEncoder, OneHotEncoder, TargetEncoder, HashEncoder
from analitica.som import SOMCluster

# Crear pipeline ETL
pipeline = Pipeline()

# Extraer datos
pipeline.extract_from_csv("data/input.csv")

# Transformar
pipeline.add_transformer(MinMaxScaler(columns=["edad", "ingreso"]))
pipeline.add_transformer(LabelEncoder(columns=["estado_civil"]))
pipeline.add_transformer(OneHotEncoder(columns=["ciudad"]))
pipeline.add_transformer(TargetEncoder(columns=["ocupacion"], y=target_variable))
pipeline.add_transformer(SOMCluster(map_size=(10, 10), iterations=1000))

# Cargar resultado
pipeline.load_to_csv("data/output.csv")

# Ejecutar
result = pipeline.run()
```

### CLI

```bash
# Ejecutar pipeline desde archivo de configuración
analitica run config/pipeline.yaml

# Mostrar información del dataset
analitica inspect data/input.csv

# Entrenar SOM con datos
analitica som-train --data data/input.csv --output models/som.pkl
```

## 📊 Dataset de Ejemplo

El proyecto incluye datasets de ejemplo en `data/samples/`:
- `students.csv`: Datos de estudiantes (mezcla numérico/categórico)
- `sales.csv`: Transacciones de venta
- `sensor_data.csv`: Datos de sensores IoT

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=analitica --cov-report=html

# Tests específicos
pytest tests/test_normalization.py -v
```

## 📚 Algoritmo SOM (Self-Organizing Maps)

### ¿Qué es un Mapa Autoorganizado?

Los **Self-Organizing Maps** (SOM) o **Mapas de Kohonen** son redes neuronales competitivas que reducen dimensionalidad mientras preservan topología. Ideales para:

- Visualización de datos de alta dimensión
- Clustering sin supervisión
- Detección de patrones en datos ETL
- Segmentación de clientes

### Uso Básico

```python
from analitica.som import SOMTrainer, SOMVisualizer, SOMAnalyzer

# Crear y entrenar SOM
trainer = SOMTrainer(x=15, y=15, input_len=4, sigma=2.0)
trainer.fit(data, epochs=200)

# Obtener asignaciones de cluster
assignments = trainer.transform(data)

# Visualizar U-Matrix
visualizer = SOMVisualizer(trainer)
visualizer.plot_umatrix()

# Analizar calidad
analyzer = SOMAnalyzer(trainer)
metrics = analyzer.get_metrics(data)
print(f"QE: {metrics['qe']:.4f}, TE: {metrics['te']:.4f}")
```

### Parámetros Principales

| Parámetro | Descripción | Valor Típico |
|-----------|-------------|--------------|
| `x`, `y` | Tamaño del grid | 10×10 a 30×30 |
| `input_len` | Dimensión de entrada | Features del dataset |
| `sigma` | Radio inicial vecindad | 1.0 a max(x,y)/2 |
| `learning_rate` | Tasa de aprendizaje | 0.5 → 0.01 |
| `epochs` | Iteraciones de entrenamiento | 100-500 |

### Métricas de Calidad

| Métrica | Descripción | Bueno | Malo |
|---------|-------------|--------|------|
| **QE** | Error de Cuantización | < 0.15 | > 0.3 |
| **TE** | Error Topográfico | < 0.1 | > 0.2 |
| **Coverage** | % neuronas usadas | > 70% | < 50% |

### U-Matrix

- **Colores fríos (azul)**: Neuronas similares = Clusters
- **Colores cálidos (rojo)**: Límites entre clusters

### Tutorial y Ejemplos

```bash
# Tutorial interactivo con explicaciones
python examples/som_tutorial.py

# Caso de uso: Customer Segmentation
python examples/som_customer_segmentation.py
```

## 🔬 Casos de Uso Académicos

1. **Normalización de datasets híbridos** (numérico + categórico)
2. **Clustering de segmentos de clientes**
3. **Reducción de dimensionalidad para visualización**
4. **Detección de anomalías basada en SOM**

## 📖 Roadmap

- [x] [v0.1] Estructura base del proyecto y README
- [x] [v0.2] Motor ETL core con extracción CSV
- [x] [v0.3] Normalizadores básicos (MinMax, ZScore, Robust, Log, Power)
- [x] [v0.4] Transformadores texto→número (Label, OneHot, Target, Hash)
- [x] [v0.5] **Integración SOM completa** ⭐
  - SOMTrainer con documentación extensiva
  - SOMPredictor para BMU y predicción
  - SOMVisualizer con U-Matrix y Component Planes
  - SOMAnalyzer con métricas QE y TE
  - Tutorial detallado y caso de uso
- [x] [v0.6] CLI completo
- [ ] [v0.7] Fuentes de datos adicionales (JSON, Excel)
- [ ] [v0.8] Pipeline completo ETL+SOM de extremo a extremo

## 📄 Licencia

MIT License - Ver archivo `LICENSE`

## 👥 Autores

Proyecto académico - Universidad

#!/usr/bin/env python3
"""
================================================================================
SOM TUTORIAL - Self-Organizing Maps: Conceptos y Uso
================================================================================

Este tutorial explica qué son los Mapas Autoorganizados (SOM) y cómo usarlos
en el proyecto Analitica.

OBJETIVOS DE APRENDIZAJE:
1. Entender qué es un SOM y cómo funciona
2. Conocer los parámetros principales y cómo ajustarlos
3. Saber interpretar las visualizaciones (U-Matrix)
4. Aplicar SOM a problemas reales de clustering

================================================================================
SECCIÓN 1: ¿QUÉ ES UN MAPA AUTOORGANIZADO (SOM)?
================================================================================

Un Self-Organizing Map (SOM) o Mapa Autoorganizado es un algoritmo de
aprendizaje NO SUPERVISADO (unsupervised learning) propuesto por Teuvo Kohonen
en 1982.

CARACTERÍSTICAS PRINCIPALES:
1. Reduce dimensionalidad: Datos N-dimensionales → Grid 2D
2. Preserva topología: Datos similares quedan cerca en el mapa
3. Clustering automático: Agrupa datos sin especificar número de clusters
4. Visualización: Permite ver estructuras en datos de alta dimensión

EJEMPLO INTUITIVO:
Imagina un mapa de una ciudad donde:
- Cada edificio = un dato
- La distancia entre edificios = similitud entre datos
- El SOM organiza los edificios de manera que edificios similares
  quedan juntos, formando vecindarios naturales.

================================================================================
SECCIÓN 2: ARQUITECTURA DEL SOM
================================================================================

Un SOM consiste en una RED DE NEURONAS en un grid 2D:

        Grid 10x10 (100 neuronas)

    Y
    ^
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][B][B][B][N][N][N][N]  ← BMU y vecinos
    |  [N][N][N][B][B][B][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    |  [N][N][N][N][N][N][N][N][N][N]
    +----------------------------------> X

    N = Neurona    B = Best Matching Unit (BMU) y vecindario

CADA NEURONA TIENE:
- Posición (x, y) en el grid
- Vector de pesos de la misma dimensión que los datos

================================================================================
SECCIÓN 3: ALGORITMO DE ENTRENAMIENTO
================================================================================

PASO 1: INICIALIZACIÓN
- Los pesos de cada neurona se inicializan (aleatorio o PCA)

PASO 2: COMPETICIÓN
Para cada dato de entrada:
1. Calcular similitud (distancia euclidiana) con TODAS las neuronas
2. Encontrar la neurona más similar → BMU (Best Matching Unit)

PASO 3: COOPERACIÓN
- El BMU "gana" y actualiza sus pesos
- Los vecinos del BMU también se actualizan (menos que el BMU)
- La influencia de los vecinos decrece con la distancia

PASO 4: ADAPTACIÓN
Nuevo peso = peso antiguo + η * h * (entrada - peso antiguo)

Donde:
- η (eta) = tasa de aprendizaje (0 < η < 1)
- h = función de vecindad (Gaussiana típicamente)
- (entrada - peso) = cuánto ajustar

FASES DE ENTRENAMIENTO:
1. ORDENAMIENTO (primeros ~25% de iteraciones):
   - Sigma grande (radio de vecindad amplio)
   - Tasa de aprendizaje alta
   - Aprende estructura GLOBAL rápidamente

2. AFINADO (últimos ~75%):
   - Sigma pequeño
   - Tasa de aprendizaje baja
   - Refina detalles LOCALES

================================================================================
SECCIÓN 4: PARÁMETROS PRINCIPALES
================================================================================

GRID SIZE (x, y):
-----------------
Determina la resolución del mapa.

Regla práctica: 5 * sqrt(N) neuronas donde N = número de muestras

Ejemplos:
- 100 muestras → grid ~10x10 (100 neuronas)
- 1000 muestras → grid ~15x15 (225 neuronas)
- 10000 muestras → grid ~20x20 (400 neuronas)

Trade-offs:
- Grid más grande: más detalles, más tiempo de entrenamiento
- Grid más pequeño: más rápido, pero puede mezclar clusters

SIGMA (radio de vecindad):
--------------------------
Controla cuántos vecinos se ven afectados por el BMU.

- Sigma grande → muchos vecinos, estructura global
- Sigma pequeño → pocos vecinos, detalles locales

Regla: sigma ≈ max(x, y) / 2

LEARNING RATE (tasa de aprendizaje):
-----------------------------------
Controla cuánto cambian los pesos en cada actualización.

- Valor inicial típico: 0.5
- Decae durante el entrenamiento hasta ~0.01

NEIGHBORHOOD FUNCTION (función de vecindad):
--------------------------------------------
Cómo la influencia decrece con la distancia:

- 'gaussian' (RECOMENDADA): Caída suave, transiciones suaves
- 'mexican_hat': Bordes más nítidos
- 'bubble': Todo-o-nada (dentro/afuera del radio)
- 'triangle': Caída lineal

================================================================================
SECCIÓN 5: MÉTRICAS DE CALIDAD
================================================================================

QUANTIZATION ERROR (QE) - Error de Cuantización:
-----------------------------------------------
Distancia promedio de cada dato a su BMU.

QE = (1/N) * Σ ||x_i - w_BMU(x_i)||

Interpretación:
- QE bajo → los datos están bien representados
- QE muy bajo → posible sobreajuste
- QE alto → el mapa no representa bien los datos

Valores típicos (datos normalizados [0,1]):
- < 0.1: Excelente
- 0.1 - 0.2: Bueno
- 0.2 - 0.3: Aceptable
- > 0.3: Mejor considerar grid más grande

TOPOGRAPHIC ERROR (TE) - Error Topológico:
-----------------------------------------
Proporción de datos donde el primer y segundo BMU NO son adyacentes.

TE = (1/N) * Σ adjacent(BMU1, BMU2) ? 0 : 1

Interpretación:
- TE bajo → la topología está bien preservada
- TE alto → el mapa no respeta el orden de los datos

Valores típicos:
- < 0.1 (10%): Excelente preservación topológica
- < 0.2: Buena
- > 0.2: La topología no está bien preservada

================================================================================
SECCIÓN 6: U-MATRIX (LA VISUALIZACIÓN CLAVE)
================================================================================

La U-Matrix (Unified Distance Matrix) es la visualización más importante.

CÓMO SE CALCULA:
Para cada neurona, calcular la distancia promedio a sus vecinos:

         [4.2]---[1.1]---[3.8]
           |       |       |
          [1.0]---[0.2]---[1.3]
           |       |       |
         [5.1]---[1.2]---[4.5]

En este ejemplo, los valores más bajos (0.2) indican donde los datos
son más similares (centro del cluster).

INTERPRETACIÓN:
- COLORES FRÍOS (azul): Neuronas similares = CLUSTERS
- COLORES CALIENTES (rojo/amarillo): Neuronas distintas = LÍMITES ENTRE CLUSTERS

EJEMPLO DE U-MATRIX:

    [Azul][Azul][Azul][Rojo][Amarillo][Rojo][Verde][Verde][Verde]
    [Azul][Azul][Azul][Rojo][Amarillo][Rojo][Verde][Verde][Verde]
    [Azul][Azul][Azul][Rojo][Amarillo][Rojo][Verde][Verde][Verde]

    → Dos clusters (azul y verde) separados por un límite (rojo/amarillo)

================================================================================
SECCIÓN 7: USO EN ANALITICA
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar módulos de Analitica
from analitica.som import SOMTrainer, SOMPredictor, SOMVisualizer, SOMAnalyzer


def create_demo_data():
    """
    Crear datos de demostración con 3 clusters naturales.

    Esta función genera datos sintéticos que forman 3 grupos
    claramente separados, ideal para demostrar el clustering del SOM.
    """
    np.random.seed(42)

    n_samples = 150

    # Cluster 1: Grupo de alta puntuación, bajo ingreso
    cluster1 = pd.DataFrame(
        {
            "score": np.random.normal(0.9, 0.05, 50),
            "income": np.random.normal(30000, 5000, 50),
            "age": np.random.normal(25, 3, 50),
            "activity": np.random.normal(80, 10, 50),
        }
    )

    # Cluster 2: Grupo de puntuación media, ingreso medio
    cluster2 = pd.DataFrame(
        {
            "score": np.random.normal(0.5, 0.1, 50),
            "income": np.random.normal(50000, 8000, 50),
            "age": np.random.normal(35, 5, 50),
            "activity": np.random.normal(50, 15, 50),
        }
    )

    # Cluster 3: Grupo de baja puntuación, alto ingreso
    cluster3 = pd.DataFrame(
        {
            "score": np.random.normal(0.2, 0.08, 50),
            "income": np.random.normal(80000, 10000, 50),
            "age": np.random.normal(45, 8, 50),
            "activity": np.random.normal(25, 12, 50),
        }
    )

    # Combinar clusters
    data = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

    # Agregar algo de ruido
    for col in data.columns:
        data[col] = data[col] + np.random.normal(0, 0.01, len(data))

    return data


def example_basic_usage():
    """
    EJEMPLO 1: Uso básico del SOM
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 1: USO BÁSICO DEL SOM")
    print("=" * 70)

    # Crear datos de demostración
    data = create_demo_data()

    print(f"\n📊 Datos: {len(data)} muestras, {len(data.columns)} características")
    print(f"   Características: {list(data.columns)}")

    # Crear y entrenar SOM
    # Regla: 5*sqrt(150) ≈ 60 neuronas → grid ~8x8
    print("\n🚀 Creando SOM 10x10...")

    trainer = SOMTrainer(
        x=10,  # 10 columnas
        y=10,  # 10 filas (100 neuronas en total)
        input_len=4,  # 4 características
        sigma=1.5,  # Radio inicial de vecindad
        learning_rate=0.5,  # Tasa de aprendizaje inicial
        random_seed=42,  # Para reproducibilidad
    )

    # Entrenar con 100 épocas
    print("⚙️ Entrenando (100 épocas)...")
    trainer.fit(data, epochs=100, verbose=True)

    # Obtener asignaciones de cluster
    print("\n📍 Obteniendo asignaciones de neuronas...")
    assignments = trainer.transform(data)

    # Añadir asignaciones a los datos originales
    result = data.copy()
    result["neuron_x"] = assignments["neuron_x"]
    result["neuron_y"] = assignments["neuron_y"]

    print(f"\n✅ Muestras asignadas a neuronas:")
    print(result[["score", "income", "neuron_x", "neuron_y"]].head(10))

    return trainer, data


def example_with_visualization(trainer, data):
    """
    EJEMPLO 2: Visualización del SOM
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 2: VISUALIZACIÓN DEL SOM")
    print("=" * 70)

    # Crear visualizador
    visualizer = SOMVisualizer(trainer)

    # Plotear U-Matrix
    print("\n📈 Generando U-Matrix...")
    print("   (Los colores cálidos indican límites entre clusters)")

    fig = visualizer.plot_umatrix(show=False)
    visualizer.save_figure(fig, "som_umatrix.png")
    print("   ✓ Guardado como 'som_umatrix.png'")

    # Plotear component planes
    print("\n📊 Generando Component Planes...")
    print("   (Muestra cómo cada característica se distribuye en el mapa)")

    fig = visualizer.plot_component_planes(
        feature_names=["Score", "Income", "Age", "Activity"], show=False
    )
    visualizer.save_figure(fig, "som_components.png")
    print("   ✓ Guardado como 'som_components.png'")

    # Plotear BMU para un ejemplo
    print("\n🎯 Generando visualización de BMU...")
    sample = data.iloc[0].values
    fig = visualizer.plot_bmu(sample, data=data.values, show=False)
    visualizer.save_figure(fig, "som_bmu.png")
    print("   ✓ Guardado como 'som_bmu.png'")


def example_with_analysis(trainer, data):
    """
    EJEMPLO 3: Análisis de calidad del SOM
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 3: ANÁLISIS DE CALIDAD DEL SOM")
    print("=" * 70)

    # Crear analizador
    analyzer = SOMAnalyzer(trainer)

    # Calcular métricas
    print("\n📏 Calculando métricas de calidad...")
    metrics = analyzer.get_metrics(data.values)

    print("\n" + "-" * 40)
    print("MÉTRICAS DE CALIDAD:")
    print("-" * 40)
    print(f"  Quantization Error (QE):  {metrics['qe']:.4f}")
    print(f"  Topographic Error (TE):   {metrics['te']:.4f}")
    print(
        f"  Neuronas usadas:          {metrics['nodes_used']}/{metrics['total_nodes']}"
    )
    print(f"  Cobertura del mapa:       {metrics['coverage'] * 100:.1f}%")
    print(f"  Hits máximo por neurona:  {metrics['max_hits']}")
    print(f"  Hits promedio (usadas):   {metrics['mean_hits']:.1f}")
    print("-" * 40)

    # Interpretación
    print("\n📝 INTERPRETACIÓN:")

    qe = metrics["qe"]
    if qe < 0.1:
        print(f"  ✓ QE = {qe:.4f} (< 0.1): Excelente representación")
    elif qe < 0.2:
        print(f"  ✓ QE = {qe:.4f} (0.1-0.2): Buena representación")
    elif qe < 0.3:
        print(f"  ⚠ QE = {qe:.4f} (0.2-0.3): Aceptable")
    else:
        print(f"  ✗ QE = {qe:.4f} (> 0.3): Considere un grid más grande")

    te = metrics["te"]
    if te < 0.1:
        print(f"  ✓ TE = {te:.4f} (< 0.1): Excelente preservación topológica")
    elif te < 0.2:
        print(f"  ✓ TE = {te:.4f} (0.1-0.2): Buena preservación topológica")
    else:
        print(f"  ⚠ TE = {te:.4f} (> 0.2): La topología no está bien preservada")

    coverage = metrics["coverage"]
    if coverage > 0.7:
        print(f"  ✓ Cobertura = {coverage * 100:.1f}%: Buena utilización del mapa")
    elif coverage > 0.5:
        print(f"  ⚠ Cobertura = {coverage * 100:.1f}%: Mapa parcialmente utilizado")
    else:
        print(
            f"  ⚠ Cobertura = {coverage * 100:.1f}%: Mapa subutilizado - "
            "considere grid más pequeño"
        )


def example_clustering(trainer, data):
    """
    EJEMPLO 4: Extracción de clusters del SOM
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 4: EXTRACCIÓN DE CLUSTERS")
    print("=" * 70)

    # Obtener asignaciones
    assignments = trainer.transform(data)

    # Contar muestras por neurona
    neuron_counts = (
        assignments.groupby(["neuron_x", "neuron_y"]).size().reset_index(name="count")
    )
    neuron_counts = neuron_counts.sort_values("count", ascending=False)

    print("\n📊 Top 10 neuronas más pobladas:")
    print(neuron_counts.head(10).to_string(index=False))

    # Identificar clusters (grupos de neuronas adyacentes con alta densidad)
    # Usamos un umbral simple: neuronas con > 2 hits
    dense_neurons = neuron_counts[neuron_counts["count"] > 2]

    print(f"\n🎯 Neuronas densas (> 2 muestras): {len(dense_neurons)}")

    # Asignar cluster labels basados en la posición en el mapa
    def assign_cluster(row):
        """Asignar cluster basado en la posición en el mapa."""
        x, y = row["neuron_x"], row["neuron_y"]
        if y < 3:  # Arriba del mapa
            return "Cluster_A"
        elif y < 7:  # Medio del mapa
            return "Cluster_B"
        else:  # Abajo del mapa
            return "Cluster_C"

    assignments["cluster"] = assignments.apply(assign_cluster, axis=1)

    print("\n📈 Distribución de clusters:")
    print(assignments["cluster"].value_counts())

    # Combinar con datos originales
    result = data.copy()
    result["neuron_x"] = assignments["neuron_x"]
    result["neuron_y"] = assignments["neuron_y"]
    result["cluster"] = assignments["cluster"]

    print("\n📋 Muestra de datos con clusters:")
    print(result[["score", "income", "cluster"]].head(10))

    return result


def example_prediction(trainer):
    """
    EJEMPLO 5: Predicción con SOM entrenado
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 5: PREDICCIÓN CON SOM")
    print("=" * 70)

    # Crear predictor
    predictor = SOMPredictor(trainer)

    # Nuevo "cliente" a clasificar
    new_sample = np.array([0.85, 35000, 28, 75])  # score, income, age, activity

    print(f"\n🎯 Clasificando nueva muestra: {new_sample}")

    # Encontrar BMU
    bmu = predictor.bmu(new_sample)
    print(f"   BMU (neurona más similar): ({bmu[0]}, {bmu[1]})")

    # Calcular quantization error
    qe = predictor.quantization_error(new_sample)
    print(f"   Quantization Error: {qe:.4f}")

    if qe < 0.1:
        print("   ✓ Muestra bien representada por el mapa")
    elif qe < 0.2:
        print("   ✓ Muestra razonablemente representada")
    else:
        print("   ⚠ Muestra puede ser un outlier")

    # Predicción completa
    node_x, node_y, distance = predictor.predict(new_sample)
    print(f"\n📍 Resultado de predicción:")
    print(f"   Nodo: ({node_x}, {node_y})")
    print(f"   Distancia: {distance:.4f}")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("=" * 70)
    print(" TUTORIAL DE SELF-ORGANIZING MAPS (SOM)")
    print(" Proyecto Analitica - ETL con Mapas Autoorganizados")
    print("=" * 70)

    print("""
    Este tutorial cubre:
    1. Uso básico del SOM
    2. Visualización (U-Matrix, Component Planes)
    3. Análisis de calidad
    4. Extracción de clusters
    5. Predicción con SOM entrenado
    
    ¡Vamos!
    """)

    # Ejecutar ejemplos
    trainer, data = example_basic_usage()
    example_with_visualization(trainer, data)
    example_with_analysis(trainer, data)
    result = example_clustering(trainer, data)
    example_prediction(trainer)

    print("\n" + "=" * 70)
    print(" TUTORIAL COMPLETADO")
    print("=" * 70)
    print("""
    Archivos generados:
    - som_umatrix.png: U-Matrix mostrando estructura de clusters
    - som_components.png: Component Planes por característica
    - som_bmu.png: BMU highlighting para muestra
    
    Próximos pasos:
    - Experimenta con diferentes grid sizes
    - Ajusta los parámetros sigma y learning_rate
    - Usa tus propios datos para clustering
    
    ¡Happy mapping! 🗺️
    """)


if __name__ == "__main__":
    main()

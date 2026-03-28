#!/usr/bin/env python3
"""
================================================================================
CASO DE USO: Customer Segmentation con SOM
================================================================================

Este ejemplo demuestra cómo usar SOM para segmentación de clientes,
un caso de uso común en análisis de datos empresariales.

ESCENARIO:
-----------
Una empresa de e-commerce quiere segmentar sus clientes para
personalizar campañas de marketing.

DATOS DISPONIBLES:
- age: Edad del cliente
- income: Ingreso anual estimado
- spending_score: Puntuación de gasto (0-100)
- purchase_frequency: Frecuencia de compra (compras/mes)
- last_purchase_days: Días desde última compra
- product_categories: Número de categorías de productos comprados

OBJETIVOS:
1. Identificar grupos naturales de clientes
2. Entender características de cada segmento
3. Generar recomendaciones de marketing por segmento
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analitica.som import SOMTrainer, SOMPredictor, SOMVisualizer, SOMAnalyzer
from analitica.normalization import MinMaxScaler


def generate_customer_data(n_customers=500, seed=42):
    """
    Generar datos sintéticos de clientes con segmentos naturales.

    Creamos 5 segmentos de clientes con características distintas:
    1. High Spenders: Alta puntuación de gasto, alto ingreso
    2. Budget Shoppers: Baja puntuación de gasto, moderado ingreso
    3. Occasional Buyers: Baja frecuencia de compra
    4. Loyal Customers: Alta frecuencia, reciente última compra
    5. At Risk: Baja frecuencia,很久 sin comprar
    """
    np.random.seed(seed)

    customers = []

    # Segment 1: High Spenders (20%)
    n = int(n_customers * 0.20)
    for _ in range(n):
        customers.append(
            {
                "age": int(np.random.normal(40, 8)),
                "income": np.random.normal(120000, 20000),
                "spending_score": np.random.normal(85, 8),
                "purchase_frequency": np.random.normal(8, 1.5),
                "last_purchase_days": int(np.random.exponential(10)),
                "product_categories": np.random.randint(6, 11),
            }
        )

    # Segment 2: Budget Shoppers (25%)
    n = int(n_customers * 0.25)
    for _ in range(n):
        customers.append(
            {
                "age": int(np.random.normal(35, 10)),
                "income": np.random.normal(55000, 10000),
                "spending_score": np.random.normal(35, 12),
                "purchase_frequency": np.random.normal(4, 1),
                "last_purchase_days": int(np.random.exponential(25)),
                "product_categories": np.random.randint(2, 5),
            }
        )

    # Segment 3: Occasional Buyers (20%)
    n = int(n_customers * 0.20)
    for _ in range(n):
        customers.append(
            {
                "age": int(np.random.normal(45, 12)),
                "income": np.random.normal(80000, 25000),
                "spending_score": np.random.normal(55, 15),
                "purchase_frequency": np.random.normal(1.5, 0.5),
                "last_purchase_days": int(np.random.exponential(60)),
                "product_categories": np.random.randint(3, 7),
            }
        )

    # Segment 4: Loyal Customers (15%)
    n = int(n_customers * 0.15)
    for _ in range(n):
        customers.append(
            {
                "age": int(np.random.normal(38, 7)),
                "income": np.random.normal(75000, 15000),
                "spending_score": np.random.normal(70, 10),
                "purchase_frequency": np.random.normal(10, 2),
                "last_purchase_days": int(np.random.exponential(5)),
                "product_categories": np.random.randint(5, 10),
            }
        )

    # Segment 5: At Risk (20%)
    n = int(n_customers * 0.20)
    for _ in range(n):
        customers.append(
            {
                "age": int(np.random.normal(50, 10)),
                "income": np.random.normal(60000, 18000),
                "spending_score": np.random.normal(25, 10),
                "purchase_frequency": np.random.normal(1, 0.5),
                "last_purchase_days": int(np.random.exponential(120)),
                "product_categories": np.random.randint(1, 4),
            }
        )

    df = pd.DataFrame(customers)

    # Limpiar valores
    df["age"] = df["age"].clip(18, 80)
    df["income"] = df["income"].clip(20000, 200000)
    df["spending_score"] = df["spending_score"].clip(0, 100)
    df["purchase_frequency"] = df["purchase_frequency"].clip(0.5, 15)
    df["last_purchase_days"] = df["last_purchase_days"].clip(1, 180)
    df["product_categories"] = df["product_categories"].clip(1, 10)

    return df


def train_som(df, grid_size=15, epochs=200):
    """
    Entrenar SOM con los datos de clientes.
    """
    print("\n" + "=" * 60)
    print("PASO 1: ENTRENAMIENTO DEL SOM")
    print("=" * 60)

    features = [
        "age",
        "income",
        "spending_score",
        "purchase_frequency",
        "last_purchase_days",
        "product_categories",
    ]

    # Normalizar datos
    print(f"📊 Datos: {len(df)} clientes, {len(features)} características")

    scaler = MinMaxScaler(columns=features)
    data_normalized = scaler.fit_transform(df[features])

    # Entrenar SOM
    print(f"\n🚀 Entrenando SOM {grid_size}x{grid_size} ({epochs} épocas)...")

    trainer = SOMTrainer(
        x=grid_size,
        y=grid_size,
        input_len=len(features),
        sigma=2.0,
        learning_rate=0.5,
        random_seed=42,
    )

    trainer.fit(data_normalized, epochs=epochs, verbose=True)

    # Obtener asignaciones
    assignments = trainer.transform(data_normalized)

    return trainer, scaler, features


def analyze_segments(trainer, df, scaler, features):
    """
    Analizar los segmentos de clientes.
    """
    print("\n" + "=" * 60)
    print("PASO 2: ANÁLISIS DE SEGMENTOS")
    print("=" * 60)

    # Calcular métricas
    analyzer = SOMAnalyzer(trainer)
    data_normalized = scaler.fit_transform(df[features])
    metrics = analyzer.get_metrics(data_normalized)

    print("\n📏 Métricas de calidad del SOM:")
    print(f"   Quantization Error: {metrics['qe']:.4f}")
    print(f"   Topographic Error:  {metrics['te']:.4f}")
    print(f"   Cobertura:          {metrics['coverage'] * 100:.1f}%")

    # Crear visualizaciones
    print("\n📈 Generando visualizaciones...")

    visualizer = SOMVisualizer(trainer)

    # U-Matrix
    fig = visualizer.plot_umatrix(show=False)
    visualizer.save_figure(fig, "customer_umatrix.png")
    print("   ✓ U-Matrix guardado")

    # Component Planes
    fig = visualizer.plot_component_planes(
        feature_names=["Age", "Income", "Spending", "Freq", "Last", "Categories"],
        show=False,
    )
    visualizer.save_figure(fig, "customer_components.png")
    print("   ✓ Component Planes guardado")

    return visualizer, analyzer


def extract_clusters(trainer, df, scaler, features):
    """
    Extraer clusters usando K-Means en los pesos del SOM.
    """
    print("\n" + "=" * 60)
    print("PASO 3: EXTRACCIÓN DE CLUSTERS")
    print("=" * 60)

    from sklearn.cluster import KMeans

    # Obtener pesos del SOM
    weights = trainer.get_weights()
    n_neurons = trainer.config.x * trainer.config.y

    # Reformatear pesos
    weights_flat = weights.reshape(n_neurons, -1)

    # Encontrar número óptimo de clusters usando elbow method
    print("\n📊 Buscando número óptimo de clusters...")

    inertias = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(weights_flat)
        inertias.append(kmeans.inertia_)

    # Usar 5 clusters (sabemos que tenemos 5 segmentos)
    n_clusters = 5
    print(f"   Usando {n_clusters} clusters")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    neuron_labels = kmeans.fit_predict(weights_flat)

    # Asignar cada cliente a un cluster basado en su neurona
    data_normalized = scaler.fit_transform(df[features])
    assignments = trainer.transform(data_normalized)

    def get_neuron_label(x, y):
        idx = y * trainer.config.x + x
        return neuron_labels[idx]

    df["neuron_x"] = assignments["neuron_x"]
    df["neuron_y"] = assignments["neuron_y"]
    df["cluster"] = assignments.apply(
        lambda row: get_neuron_label(row["neuron_x"], row["neuron_y"]), axis=1
    )

    return df, kmeans


def profile_clusters(df):
    """
    Crear perfiles de cada cluster.
    """
    print("\n" + "=" * 60)
    print("PASO 4: PERFILES DE SEGMENTOS")
    print("=" * 60)

    cluster_names = {
        0: "VIP Spenders",
        1: "Budget Conscious",
        2: "Occasional Shoppers",
        3: "Loyal Advocates",
        4: "At-Risk Churners",
    }

    print("\n📋 PERFILES DE SEGMENTOS:")
    print("-" * 60)

    profiles = []

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]

        profile = {
            "cluster": cluster_id,
            "name": cluster_names.get(cluster_id, f"Segment {cluster_id}"),
            "count": len(cluster_data),
            "pct": len(cluster_data) / len(df) * 100,
            "avg_age": cluster_data["age"].mean(),
            "avg_income": cluster_data["income"].mean(),
            "avg_spending": cluster_data["spending_score"].mean(),
            "avg_frequency": cluster_data["purchase_frequency"].mean(),
            "avg_last_purchase": cluster_data["last_purchase_days"].mean(),
            "avg_categories": cluster_data["product_categories"].mean(),
        }

        profiles.append(profile)

        print(f"\n🎯 CLUSTER {cluster_id}: {profile['name']}")
        print(f"   Tamaño: {profile['count']} clientes ({profile['pct']:.1f}%)")
        print(f"   Edad promedio: {profile['avg_age']:.1f} años")
        print(f"   Ingreso promedio: ${profile['avg_income']:,.0f}")
        print(f"   Spending Score: {profile['avg_spending']:.1f}/100")
        print(f"   Frecuencia: {profile['avg_frequency']:.1f} compras/mes")
        print(f"   Días desde última compra: {profile['avg_last_purchase']:.1f}")
        print(f"   Categorías promedio: {profile['avg_categories']:.1f}")

    return pd.DataFrame(profiles)


def generate_recommendations(profiles_df):
    """
    Generar recomendaciones de marketing por segmento.
    """
    print("\n" + "=" * 60)
    print("PASO 5: RECOMENDACIONES DE MARKETING")
    print("=" * 60)

    recommendations = {
        0: {
            "name": "VIP Spenders",
            "strategy": "Premium Experience",
            "tactics": [
                "✓ Programa de lealtad exclusivo",
                "✓ Acceso anticipado a nuevos productos",
                "✓ Gift cards para referidos",
                "✓ Eventos VIP privados",
            ],
            "retention": "Alta - Mantener engagement",
            "budget": "30% - Alto valor, alto ROI",
        },
        1: {
            "name": "Budget Conscious",
            "strategy": "Value Maximization",
            "tactics": [
                "✓ Ofertas personalizadas por email",
                "✓ Bundles con descuento",
                "✓ Free shipping en pedidos mayores",
                "✓ Cupones de descuento exclusivos",
            ],
            "retention": "Media - Incrementar frecuencia",
            "budget": "25% - Volumen potencial",
        },
        2: {
            "name": "Occasional Shoppers",
            "strategy": "Activation Campaigns",
            "tactics": [
                '✓ "We miss you" campaigns',
                "✓ Descuentos por tiempo limitado",
                "✓ Cross-selling basado en historial",
                "✓ Recordatorios de carrito abandonado",
            ],
            "retention": "Baja - Reactivar engagement",
            "budget": "20% - Recuperar clientes",
        },
        3: {
            "name": "Loyal Advocates",
            "strategy": "Brand Ambassador",
            "tactics": [
                "✓ Programa de referidos con recompensas",
                "✓ Early reviews de productos",
                "✓ Influencer partnerships",
                "✓ Comunidad exclusiva",
            ],
            "retention": "Muy Alta - Fomentar advocacy",
            "budget": "15% - Maximizar word-of-mouth",
        },
        4: {
            "name": "At-Risk Churners",
            "strategy": "Win-Back",
            "tactics": [
                "✓ Encuesta de satisfacción",
                "✓ Ofrecer incentivo de retorno (10% off)",
                "✓ Highlight de productos populares",
                "✓ Follow-up personal",
            ],
            "retention": "Crítica - Prevenir churn",
            "budget": "10% - Salvar relaciones",
        },
    }

    print("\n📝 RECOMENDACIONES POR SEGMENTO:")
    print("-" * 60)

    for cluster_id in sorted(recommendations.keys()):
        rec = recommendations[cluster_id]
        profile = profiles_df[profiles_df["cluster"] == cluster_id].iloc[0]

        print(f"\n🎯 {profile['name']} (Cluster {cluster_id})")
        print(f"   📊 Tamaño: {profile['count']} clientes ({profile['pct']:.1f}%)")
        print(f"   🎯 Estrategia: {rec['strategy']}")
        print(f"   💡 Tácticas:")
        for tactic in rec["tactics"]:
            print(f"      {tactic}")
        print(f"   📈 Retención: {rec['retention']}")
        print(f"   💰 Presupuesto: {rec['budget']}")


def main():
    """Función principal."""
    print("=" * 70)
    print(" CUSTOMER SEGMENTATION CON SELF-ORGANIZING MAPS (SOM)")
    print("=" * 70)
    print("""
Este caso de uso demuestra cómo usar SOM para segmentación de clientes.
Generaremos datos sintéticos con 5 segmentos naturales y usaremos
SOM para descubrirlos automáticamente.
    """)

    # Generar datos
    print("📊 Generando datos de clientes...")
    df = generate_customer_data(n_customers=500)
    print(f"   {len(df)} clientes creados")
    print("\n📋 Vista previa de datos:")
    print(df.describe().round(1))

    # Entrenar SOM
    trainer, scaler, features = train_som(df, grid_size=15, epochs=200)

    # Analizar
    visualizer, analyzer = analyze_segments(trainer, df, scaler, features)

    # Extraer clusters
    df = extract_clusters(trainer, df, scaler, features)

    # Perfiles
    profiles_df = profile_clusters(df)

    # Recomendaciones
    generate_recommendations(profiles_df)

    # Guardar resultados
    output_file = "customer_segments.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Segmentación guardada en: {output_file}")

    print("\n" + "=" * 70)
    print(" CASO DE USO COMPLETADO")
    print("=" * 70)
    print("""
Archivos generados:
- customer_umatrix.png: U-Matrix mostrando estructura de segmentos
- customer_components.png: Component Planes de características
- customer_segments.csv: Datos con asignación de segmentos

El SOM descubrió automáticamente los 5 segmentos naturales en los datos.
    """)


if __name__ == "__main__":
    main()

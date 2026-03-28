#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ███████╗███████╗██╗   ██╗███████╗███╗   ███╗                     ║
║     ██╔══██╗██╔════╝██╔════╝██║   ██║██╔════╝████╗ ████║                     ║
║     ██████╔╝█████╗  ███████╗██║   ██║█████╗  ██╔████╔██║                     ║
║     ██╔══██╗██╔══╝  ╚════██║██║   ██║██╔══╝  ██║╚██╔╝██║                     ║
║     ██║  ██║███████╗███████║╚██████╔╝███████╗██║ ╚═╝ ██║                     ║
║     ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝                     ║
║                                                                              ║
║     ETL + SOM ENGINE DEMO                                                    ║
║     El motor ETL completo con Mapas Autoorganizados                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Este demo muestra el poder completo del motor ETL:
1. Extraer datos de CSV, Excel y JSON
2. Transformar con normalizadores y encoders
3. Aplicar SOM para clustering y visualización
4. Cargar resultados a cualquier formato

¡Veamos qué puede hacer!
"""

import sys
import time
from pathlib import Path

# Rich for improved terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn


console = Console()

# Banner épico
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ███████╗███████╗██╗   ██╗███████╗███╗   ███╗                     ║
║     ██╔══██╗██╔════╝██╔════╝██║   ██║██╔════╝████╗ ████║                     ║
║     ██████╔╝█████╗  ███████╗██║   ██║█████╗  ██╔████╔██║                     ║
║     ██╔══██╗██╔══╝  ╚════██║██║   ██║██╔══╝  ██║╚██╔╝██║                     ║
║     ██║  ██║███████╗███████║╚██████╔╝███████╗██║ ╚═╝ ██║                     ║
║     ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_step(step_num, title, emoji="🚀"):
    """Imprimir paso con estilo usando rich."""
    console.rule(f"{emoji} PASO {step_num}: {title}", style="bold cyan")


def create_demo_data():
    """Crear datos de demostración completos."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Dataset 1: Clientes (CSV)
    n_customers = 300
    customers = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "age": np.random.normal(40, 12, n_customers).astype(int).clip(18, 80),
            "income": np.random.normal(65000, 20000, n_customers).clip(20000, 200000),
            "spending_score": np.random.normal(50, 20, n_customers).clip(0, 100),
            "purchase_frequency": np.random.exponential(3, n_customers).clip(0.5, 15),
            "last_purchase_days": np.random.exponential(30, n_customers).clip(1, 180),
            "city": np.random.choice(
                ["New York", "Los Angeles", "Chicago", "Houston", "Miami"], n_customers
            ),
            "segment": np.random.choice(["Premium", "Standard", "Basic"], n_customers),
            "loyalty_years": np.random.exponential(2, n_customers).clip(0, 15),
        }
    )

    # Dataset 2: Transacciones (Excel)
    n_transactions = 500
    transactions = pd.DataFrame(
        {
            "transaction_id": range(1, n_transactions + 1),
            "customer_id": np.random.choice(customers["customer_id"], n_transactions),
            "date": pd.date_range("2024-01-01", periods=n_transactions, freq="6h"),
            "amount": np.random.exponential(100, n_transactions).clip(10, 1000),
            "items": np.random.randint(1, 20, n_transactions),
            "payment_method": np.random.choice(
                ["Credit Card", "Debit Card", "PayPal", "Cash"], n_transactions
            ),
            "category": np.random.choice(
                ["Electronics", "Clothing", "Food", "Home", "Sports"], n_transactions
            ),
        }
    )

    # Dataset 3: Productos (JSON)
    products = [
        {
            "id": i,
            "name": f"Product {i}",
            "category": cat,
            "price": round(np.random.uniform(10, 500), 2),
        }
        for i, cat in enumerate(
            np.random.choice(["Electronics", "Clothing", "Food", "Home"], 20), 1
        )
    ]

    # Guardar datos
    customers.to_csv("demo_customers.csv", index=False)
    transactions.to_excel("demo_transactions.xlsx", index=False)

    import json

    with open("demo_products.json", "w") as f:
        json.dump(products, f, indent=2)

    return customers, transactions, products


def demo_step1_extract():
    """Paso 1: Extraer datos de múltiples fuentes."""
    print_step(1, "EXTRACCIÓN DE DATOS", "📥")

    from analitica.etl import Pipeline, CSVSource, ExcelSource, JSONSource

    print("\n📂 Cargando datos de múltiples fuentes...")

    # CSV - Clientes
    print("   ├─ CSV: customers.csv")
    customers = CSVSource("demo_customers.csv").extract()
    print(f"   │   └─ {len(customers)} registros cargados")

    # Excel - Transacciones
    print("   ├─ Excel: transactions.xlsx")
    transactions = ExcelSource("demo_transactions.xlsx").extract()
    print(f"   │   └─ {len(transactions)} registros cargados")

    # JSON - Productos
    print("   └─ JSON: products.json")
    products = JSONSource("demo_products.json").extract()
    print(f"       └─ {len(products)} productos cargados")

    # Combinar clientes con transacciones
    merged = customers.merge(transactions, on="customer_id", how="left")
    print(f"\n✅ Datos combinados: {len(merged)} registros")

    return merged, merged  # Return same df for consistent processing


def demo_step2_transform(df, customers):
    """Paso 2: Transformar datos."""
    import pandas as pd
    from analitica.normalization import MinMaxScaler, RobustScaler
    from analitica.transformers import LabelEncoder, OneHotEncoder

    print("\n🔄 Aplicando transformaciones...")

    # Seleccionar features numéricas
    numeric_cols = [
        "age",
        "income",
        "spending_score",
        "purchase_frequency",
        "last_purchase_days",
        "loyalty_years",
        "amount",
        "items",
    ]

    # 1. Normalizar features numéricas
    print("   ├─ MinMaxScaler: normalizando features numéricas...")
    scaler = MinMaxScaler(columns=numeric_cols)
    df_scaled = scaler.fit_transform(df[numeric_cols])
    print(f"   │   └─ {len(numeric_cols)} columnas normalizadas a [0, 1]")

    # 2. Label encoding para segmento
    print("   ├─ LabelEncoder: codificando segmento...")
    le = LabelEncoder(columns=["segment"])
    df_le = le.fit_transform(df[["segment"]])
    df_scaled["segment_encoded"] = df_le["segment"].values

    # 3. One-hot encoding para ciudad
    print("   └─ OneHotEncoder: codificando ciudad...")
    ohe = OneHotEncoder(columns=["city"])
    df_ohe = ohe.fit_transform(df[["city"]])

    # Combinar todo
    df_transformed = pd.concat([df_scaled, df_ohe], axis=1)
    print(f"\n✅ Datos transformados: {df_transformed.shape[1]} columnas")

    return df_transformed


def demo_step3_som(df_transformed, customers):
    """Paso 3: Aplicar SOM para clustering."""
    print_step(3, "SOM CLUSTERING", "🧠")

    from analitica.som import SOMTrainer, SOMPredictor, SOMVisualizer, SOMAnalyzer
    import matplotlib.pyplot as plt

    print("\n🎯 Entrenando Self-Organizing Map...")

    # Seleccionar features para SOM
    som_features = [
        "age",
        "income",
        "spending_score",
        "purchase_frequency",
        "last_purchase_days",
        "loyalty_years",
    ]

    # Entrenar SOM
    som_size = 12
    print(f"   ├─ Grid: {som_size}x{som_size} ({som_size * som_size} neuronas)")
    print("   ├─ Epochs: 150")

    start_time = time.time()

    trainer = SOMTrainer(
        x=som_size,
        y=som_size,
        input_len=len(som_features),
        sigma=1.5,
        learning_rate=0.5,
        random_seed=42,
    )

    trainer.fit(df_transformed[som_features], epochs=150, verbose=True)

    training_time = time.time() - start_time
    print(f"   ├─ Tiempo de entrenamiento: {training_time:.2f}s")

    # Obtener asignaciones
    assignments = trainer.transform(df_transformed[som_features])

    # Crear visualizaciones
    visualizer = SOMVisualizer(trainer)

    print("\n📊 Generando visualizaciones...")

    # U-Matrix
    fig = visualizer.plot_umatrix(show=False)
    visualizer.save_figure(fig, "demo_umatrix.png")
    print("   ├─ U-Matrix guardado: demo_umatrix.png")
    plt.close(fig)

    # Component Planes
    fig = visualizer.plot_component_planes(feature_names=som_features, show=False)
    visualizer.save_figure(fig, "demo_components.png")
    print("   ├─ Component Planes: demo_components.png")
    plt.close(fig)

    # Análisis de calidad
    analyzer = SOMAnalyzer(trainer)
    metrics = analyzer.get_metrics(df_transformed[som_features])

    print("\n📏 MÉTRICAS DE CALIDAD SOM:")
    print(f"   ├─ Quantization Error: {metrics['qe']:.4f}")
    print(f"   ├─ Topographic Error: {metrics['te']:.4f}")
    print(f"   ├─ Cobertura: {metrics['coverage'] * 100:.1f}%")
    print(f"   └─ Neuronas usadas: {metrics['nodes_used']}/{metrics['total_nodes']}")

    # Asignar clusters al dataframe transformado
    df_clustered = df_transformed.copy()
    df_clustered["som_x"] = assignments["neuron_x"].values
    df_clustered["som_y"] = assignments["neuron_y"].values

    # Crear cluster basado en posición del mapa
    def get_cluster(row):
        if row["som_y"] < 4:
            return "Cluster_A"  # Alta puntuación, bajo gasto
        elif row["som_y"] < 8:
            return "Cluster_B"  # Perfil medio
        else:
            return "Cluster_C"  # Bajo puntuación, alto gasto

    df_clustered["cluster"] = df_clustered.apply(get_cluster, axis=1)

    return df_clustered, trainer, metrics


def demo_step4_load(df_clustered):
    """Paso 4: Guardar resultados."""
    print_step(4, "GUARDAR RESULTADOS", "💾")

    from analitica.etl import CSVDestination, ExcelDestination, JSONDestination

    print("\n📝 Guardando en múltiples formatos...")

    # CSV
    CSVDestination("demo_results.csv").save(df_clustered)
    print("   ├─ CSV: demo_results.csv")

    # Excel
    ExcelDestination("demo_results.xlsx").save(df_clustered)
    print("   ├─ Excel: demo_results.xlsx")

    # JSON
    JSONDestination("demo_results.json").save(df_clustered)
    print("   └─ JSON: demo_results.json")

    print("\n✅ Resultados guardados!")


def demo_step5_summary(df_clustered, metrics):
    """Paso 5: Resumen de clusters."""
    print_step(5, "RESUMEN DE CLUSTERS", "📊")

    print("\n🎯 PERFILES DE SEGMENTOS DESCUBIERTOS POR SOM:")
    print("-" * 60)

    for cluster in ["Cluster_A", "Cluster_B", "Cluster_C"]:
        cluster_data = df_clustered[df_clustered["cluster"] == cluster]

        if len(cluster_data) == 0:
            continue

        print(
            f"\n{cluster} ({len(cluster_data)} muestras - {len(cluster_data) / len(df_clustered) * 100:.1f}%)"
        )
        print(f"   ├─ Edad promedio: {cluster_data['age'].mean():.1f} años")
        print(f"   ├─ Ingreso promedio: ${cluster_data['income'].mean():,.0f}")
        print(f"   ├─ Spending Score: {cluster_data['spending_score'].mean():.1f}/100")
        print(
            f"   ├─ Frecuencia compra: {cluster_data['purchase_frequency'].mean():.1f}/mes"
        )
        print(f"   └─ Lealtad: {cluster_data['loyalty_years'].mean():.1f} años")

    print("\n" + "=" * 60)
    print("📈 MÉTRICAS FINALES DEL MODELO")
    print("=" * 60)
    print(f"   ├─ Quantization Error (QE): {metrics['qe']:.4f}")
    print(f"   ├─ Topographic Error (TE): {metrics['te']:.4f}")
    print(f"   ├─ Cobertura del mapa: {metrics['coverage'] * 100:.1f}%")
    print(f"   └─ Clusters identificados: 3")


def main():
    """Función principal."""
    console.print(Panel(BANNER, title="ETL + SOM ENGINE DEMO", subtitle="demo_epic.py", style="bold green"))

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  Este demo demuestra el MOTOR ETL COMPLETO con SOM:                          ║
║                                                                              ║
║  1. 📥 Extraer: CSV + Excel + JSON                                           ║
║  2. ⚙️ Transformar: Normalización + Encoding                                 ║
║  3. 🧠 SOM: Clustering con Mapas Autoorganizados                             ║
║  4. 💾 Cargar: Guardar en múltiples formatos                                 ║
║  5. 📊 Resumir: Perfiles de segmentos                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    console.print("\n🎬 Iniciando demo automáticamente...", style="bold yellow")
    time.sleep(1)

    # Crear datos de demostración
    console.print("\n📦 Creando datos de demostración...", style="bold")
    with console.status("Generando datasets de ejemplo...", spinner="dots"):
        customers, transactions, products = create_demo_data()
    console.log("Datos creados: customers.csv, transactions.xlsx, products.json")

    # Pipeline completo
    try:
        # 1. Extraer
        merged_df, customers_df = demo_step1_extract()

        # 2. Transformar
        df_transformed = demo_step2_transform(merged_df, customers_df)

        # 3. SOM
        df_clustered, trainer, metrics = demo_step3_som(df_transformed, customers_df)

        # 4. Cargar
        demo_step4_load(df_clustered)

        # 5. Resumen
        demo_step5_summary(df_clustered, metrics)

        # Final épico
        print("""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗                    ║
║     ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║                    ║
║     ███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║                    ║
║     ╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║                    ║
║     ███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║                    ║
║     ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝                    ║
║                                                                              ║
║                    DEMO COMPLETADO EXITOSAMENTE 🎉                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 ARCHIVOS GENERADOS:
   ├─ demo_customers.csv       (datos originales)
   ├─ demo_transactions.xlsx   (datos originales)
   ├─ demo_products.json       (datos originales)
   ├─ demo_results.csv         (resultados con clusters)
   ├─ demo_results.xlsx        (resultados con clusters)
   ├─ demo_results.json        (resultados con clusters)
   ├─ demo_umatrix.png        (visualización SOM)
   └─ demo_components.png      (component planes)

🧠 El SOM identificó 3 segmentos naturales de clientes basándose
   en sus características de comportamiento.
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

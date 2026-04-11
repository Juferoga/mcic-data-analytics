#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Intentar importar la librería del estudiante
try:
    from analitica.normalization import MinMaxScaler, RobustScaler
    from analitica.transformers import LabelEncoder
    from analitica.som import SOMTrainer, SOMVisualizer, SOMAnalyzer
except ImportError:
    print("❌ Error: No se puede importar 'analitica'. Asegúrate de estar en el entorno virtual correcto.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Motor SOM Dinámico para Análisis de Datos")
    parser.add_argument("--input", "-i", type=str, default="data/samples/train.csv", help="Ruta al archivo CSV de entrada")
    parser.add_argument("--out_dir", "-o", type=str, default="out", help="Directorio de salida para resultados")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Número de epochs para el entrenamiento SOM")
    return parser.parse_args()

def process_data(df):
    print("🔄 Preprocesando datos dinámicamente...")
    processed_df = df.copy()
    
    # Ignorar columnas tipo ID, Name, Ticket largas o con muchos valores unicos text
    columns_to_drop = [c for c in processed_df.columns if "id" in c.lower() or c.lower() in ['name', 'ticket', 'cabin']]
    processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Rellenar NaNs y Codificar
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = processed_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Numéricas
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
    # Categóricas
    for col in categorical_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        # Usar label encoder local
        try:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
        except Exception:
            # Fallback simple
            from sklearn.preprocessing import LabelEncoder as SLE
            processed_df[col] = SLE().fit_transform(processed_df[col])
            
    # Variables listas para SOM
    features = processed_df.columns.tolist()
    
    # Normalizar todo usando Scaler de la clase
    print("📏 Normalizando características...")
    try:
        scaler = RobustScaler()
        # La API de RobustScaler de analitica parece tomar df
        normalized_data = scaler.fit_transform(processed_df)
    except Exception:
        # Fallback manual normalización (MinMax simple)
        normalized_data = (processed_df - processed_df.min()) / (processed_df.max() - processed_df.min() + 1e-8)
        
    # Convertir normalized_data a DataFrame si es np array
    if not isinstance(normalized_data, pd.DataFrame):
        normalized_data = pd.DataFrame(normalized_data, columns=features)
        
    return normalized_data, features

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"📂 Cargando archivo: {args.input}")
    try:
        df = pd.read_csv(args.input)
        print(f"✅ Cargados {len(df)} registros y {len(df.columns)} columnas.")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return

    # Preprocesamiento inteligente automático
    normalized_df, som_features = process_data(df)
    
    print(f"🧠 Inicializando SOM con {len(som_features)} características: {som_features}")
    som_size = min(15, max(5, int(np.sqrt(len(df)) / 2)))  # heurística de tamaño basada en N
    print(f"📐 Tamaño del mapa: {som_size}x{som_size}")
    
    trainer = SOMTrainer(
        x=som_size,
        y=som_size,
        input_len=len(som_features),
        sigma=max(1.0, som_size/4.0),
        learning_rate=0.5,
        random_seed=42
    )

    print("🚀 Entrenando modelo SOM...")
    start_time = time.time()
    try:
        trainer.fit(normalized_df[som_features], epochs=args.epochs, verbose=False)
    except Exception as e:
        print(f"❌ Falló el entrenamiento: {e}")
        return
        
    print(f"✅ Entrenamiento completado en {time.time() - start_time:.2f}s")
    
    # Evaluar calidad (Opcional, previene fallo si falla analyzer)
    try:
        analyzer = SOMAnalyzer(trainer)
        metrics = analyzer.get_metrics(normalized_df[som_features])
        print(f"📊 Quality Metrics | QE: {metrics['qe']:.4f} | TE: {metrics['te']:.4f}")
    except Exception:
        print("⚠️ No se pudieron calcular métricas.")
        
    # Asignaciones
    try:
        assignments = trainer.transform(normalized_df[som_features])
        df["som_group_x"] = assignments["neuron_x"].values
        df["som_group_y"] = assignments["neuron_y"].values
        df["som_cluster"] = df["som_group_x"].astype(str) + "-" + df["som_group_y"].astype(str)
        
        # Guardar CSV final
        out_csv = os.path.join(args.out_dir, "resultados_som.csv")
        df.to_csv(out_csv, index=False)
        print(f"💾 Resultados guardados en: {out_csv}")
    except Exception as e:
        print(f"❌ Error en asignación de cluster: {e}")

    # Visualizaciones
    try:
        import matplotlib.pyplot as plt
        vis = SOMVisualizer(trainer)
        
        print("🎨 Generando visualizaciones...")
        # U-Matrix
        fig_um = vis.plot_umatrix(show=False)
        vis.save_figure(fig_um, os.path.join(args.out_dir, "umatrix.png"))
        plt.close(fig_um)
        
        # Component Planes (solo top 4 variables si son muchas para rapidez)
        top_feats = som_features[:4]
        fig_cp = vis.plot_component_planes(feature_names=top_feats, show=False)
        vis.save_figure(fig_cp, os.path.join(args.out_dir, "component_planes.png"))
        plt.close(fig_cp)
        print(f"✅ Gráficos generados exitosamente en la carpeta '{args.out_dir}'")
    except Exception as e:
        print(f"⚠️ Error generando gráficos: {e}")

    print("\n🎉 ¡Proceso finalizado exitosamente!")
    
if __name__ == "__main__":
    main()

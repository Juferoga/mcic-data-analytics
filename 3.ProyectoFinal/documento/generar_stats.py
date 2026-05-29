import matplotlib.pyplot as plt
import numpy as np

# Datos
labels = ['Nuevos casos\ndiagnosticados', 'Muertes\nregistradas']
valores = [2.3, 0.685]  # millones

fig, ax = plt.subplots(figsize=(6, 4.5))

colors = ['#e74c3c', '#c0392b']
bars = ax.bar(labels, valores, color=colors, width=0.55, edgecolor='black', linewidth=1.2)

# Agregar valores encima de las barras
for bar, val in zip(bars, valores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.04,
            f'{val:.1f} millones' if val >= 1 else f'{int(val*1000):,}',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')

# Ajustes
ax.set_ylabel('Millones de personas', fontsize=12, fontweight='bold')
ax.set_title('Cáncer de mama a nivel global — Año 2020\n(Fuente: GLOBOCAN / Sung et al., 2021)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 3.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=10)

# Grid horizontal
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('images/fig_global_stats_2020.png', dpi=300, bbox_inches='tight')
print("Imagen guardada: images/fig_global_stats_2020.png")

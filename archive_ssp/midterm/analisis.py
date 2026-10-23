"""
analisis.py
============
Dos analisis sobre results_formants_clean.csv:

  A) ANALISIS GLOBAL — comparacion Norte vs Sur en el espacio F1/F2
     - Estadisticos descriptivos por dialecto
     - Test de Welch (t-test) para F1 y F2
     - Plot: dispersion F1 vs F2 coloreado por dialecto + elipses de confianza

  B) CLASIFICACION K-MEANS — 5 clusters en el espacio F1/F2
     - K-Means con k=5 (una por vocal del espanol)
     - Los centroides se interpretan como /a/ /e/ /i/ /o/ /u/
       segun su posicion en el espacio formántico
     - Plot: espacio vocalico con clusters + etiquetas interpretadas
     - Tabla: distribucion de clusters por dialecto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── CONFIGURACION ─────────────────────────────────────────────────────────
SEMILLA = 42          # reproducibilidad
N_CLUSTERS = 5        # cinco vocales del espanol

# ── CARGAR DATOS ──────────────────────────────────────────────────────────
df = pd.read_csv('results_formants_clean.csv')
print(f"Datos cargados: {len(df)} tokens ({df['dialecto'].value_counts().to_dict()})\n")


# ═══════════════════════════════════════════════════════════════════════════
# A) ANALISIS GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("A) ANALISIS GLOBAL — Norte vs Sur")
print("=" * 55)

norte = df[df['dialecto'] == 'norte']
sur   = df[df['dialecto'] == 'sur']

for formante in ['F1_Hz', 'F2_Hz']:
    t, p = stats.ttest_ind(norte[formante], sur[formante], equal_var=False)
    print(f"\n{formante}:")
    print(f"  Norte: M={norte[formante].mean():.1f} Hz, SD={norte[formante].std():.1f}")
    print(f"  Sur:   M={sur[formante].mean():.1f} Hz, SD={sur[formante].std():.1f}")
    print(f"  Welch t={t:.3f}, p={p:.4f} {'*** significativo' if p < 0.05 else '(no significativo)'}")

# Cohen's d
def cohens_d(a, b):
    pooled_sd = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd

d_f1 = cohens_d(norte['F1_Hz'], sur['F1_Hz'])
d_f2 = cohens_d(norte['F2_Hz'], sur['F2_Hz'])
print(f"\nTamaño del efecto (Cohen's d):")
print(f"  F1: d={d_f1:.3f}")
print(f"  F2: d={d_f2:.3f}")


# ── Plot A: dispersion Norte vs Sur ───────────────────────────────────────

def elipse_confianza(ax, x, y, color, n_std=1.5):
    """Dibuja una elipse de confianza alrededor de una nube de puntos."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    orden = vals.argsort()[::-1]
    vals, vecs = vals[orden], vecs[:, orden]
    angulo = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ancho, alto = 2 * n_std * np.sqrt(vals)
    elipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                     width=ancho, height=alto, angle=angulo,
                     edgecolor=color, fc='none', lw=2, linestyle='--')
    ax.add_patch(elipse)

fig, ax = plt.subplots(figsize=(9, 7))

colores = {'norte': '#2166ac', 'sur': '#d6604d'}
for dialecto, grupo in df.groupby('dialecto'):
    ax.scatter(grupo['F2_Hz'], grupo['F1_Hz'],
               c=colores[dialecto], alpha=0.35, s=25, label=dialecto.capitalize())
    elipse_confianza(ax, grupo['F2_Hz'].values, grupo['F1_Hz'].values, colores[dialecto])
    # centroide
    ax.scatter(grupo['F2_Hz'].mean(), grupo['F1_Hz'].mean(),
               c=colores[dialecto], s=120, marker='*', zorder=5)

# Ejes invertidos: convencion fonetica
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel('F2 (Hz)', fontsize=12)
ax.set_ylabel('F1 (Hz)', fontsize=12)
ax.set_title('Espacio vocálico: Norte vs Sur peninsular\n(Mozilla Common Voice, español)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_global.png', dpi=150)
plt.close()
print("\n→ Guardado: plot_global.png")


# ═══════════════════════════════════════════════════════════════════════════
# B) CLASIFICACION K-MEANS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("B) K-MEANS — 5 clusters (vocales)")
print("=" * 55)

# Normalizar antes de clustering (F1 y F2 tienen escalas similares
# pero estandarizar evita que uno domine la distancia euclidea)
X = df[['F1_Hz', 'F2_Hz']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEMILLA, n_init=20)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Centroides en Hz reales (desescalados)
centroides_hz = scaler.inverse_transform(kmeans.cluster_centers_)
cent_df = pd.DataFrame(centroides_hz, columns=['F1_Hz', 'F2_Hz'])

# Interpretar cada cluster como una vocal segun su posicion:
# F1 alto = vocal abierta (/a/), F1 bajo = cerrada (/i/ /u/)
# F2 alto = anterior (/i/ /e/), F2 bajo = posterior (/o/ /u/)
#
# Asignacion automatica por ranking:
#   /a/ = F1 mas alto
#   /i/ = F1 mas bajo + F2 mas alto
#   /u/ = F1 mas bajo + F2 mas bajo
#   /e/ = F1 medio-alto + F2 alto
#   /o/ = F1 medio + F2 bajo

cent_df['cluster'] = range(N_CLUSTERS)
cent_df = cent_df.sort_values('F1_Hz', ascending=False).reset_index(drop=True)

# /a/ = F1 maximo
# Los dos con F1 bajo: el de F2 mayor es /i/, el de F2 menor es /u/
# Los dos intermedios: el de F2 mayor es /e/, el de F2 menor es /o/
etiquetas_orden = ['a', None, None, None, None]
bajos  = cent_df.iloc[3:].sort_values('F2_Hz', ascending=False)
medios = cent_df.iloc[1:3].sort_values('F2_Hz', ascending=False)

mapa_vocal = {}
mapa_vocal[cent_df.iloc[0]['cluster']] = 'a'
mapa_vocal[medios.iloc[0]['cluster']]  = 'e'
mapa_vocal[medios.iloc[1]['cluster']]  = 'o'
mapa_vocal[bajos.iloc[0]['cluster']]   = 'i'
mapa_vocal[bajos.iloc[1]['cluster']]   = 'u'

df['vocal_kmeans'] = df['cluster'].map(mapa_vocal)

print("\nCentroides de clusters (en Hz):")
for c_id, vocal in mapa_vocal.items():
    row = cent_df[cent_df['cluster'] == c_id].iloc[0]
    print(f"  /{vocal}/  F1={row['F1_Hz']:.0f} Hz,  F2={row['F2_Hz']:.0f} Hz")

print("\nDistribucion clusters por dialecto:")
tabla = df.groupby(['dialecto', 'vocal_kmeans']).size().unstack(fill_value=0)
print(tabla)

# Porcentajes (mas util para comparar dialectos de distinto tamaño)
print("\nPorcentajes por dialecto (% dentro de cada dialecto):")
print(tabla.div(tabla.sum(axis=1), axis=0).round(3) * 100)


# ── Plot B: espacio vocalico con clusters ─────────────────────────────────

colores_vocal = {
    'a': '#e41a1c', 'e': '#ff7f00',
    'i': '#377eb8', 'o': '#4daf4a', 'u': '#984ea3'
}
marcadores = {'norte': 'o', 'sur': '^'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, dialecto in zip(axes, ['norte', 'sur']):
    sub = df[df['dialecto'] == dialecto]
    for vocal, grupo in sub.groupby('vocal_kmeans'):
        ax.scatter(grupo['F2_Hz'], grupo['F1_Hz'],
                   c=colores_vocal[vocal], alpha=0.4, s=30,
                   marker=marcadores[dialecto], label=f'/{vocal}/')

    # centroides
    for c_id, vocal in mapa_vocal.items():
        row = cent_df[cent_df['cluster'] == c_id].iloc[0]
        ax.scatter(row['F2_Hz'], row['F1_Hz'],
                   c=colores_vocal[vocal], s=200, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(f'/{vocal}/', (row['F2_Hz'], row['F1_Hz']),
                    textcoords='offset points', xytext=(8, 4), fontsize=12, fontweight='bold')

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)', fontsize=11)
    ax.set_ylabel('F1 (Hz)', fontsize=11)
    ax.set_title(f'{dialecto.capitalize()} peninsular', fontsize=12)
    ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=colores_vocal[v], label=f'/{v}/') for v in 'aeiou']
    ax.legend(handles=handles, fontsize=10)

fig.suptitle('Espacio vocálico por dialecto — clusters k-means (k=5)', fontsize=13)
plt.tight_layout()
plt.savefig('plot_kmeans.png', dpi=150)
plt.close()
print("\n→ Guardado: plot_kmeans.png")

print("\n✅  Analisis completo.")
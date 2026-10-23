import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# configuracion
SEMILLA = 42
df = pd.read_csv('results_formants_clean.csv')
norte = df[df['dialecto'] == 'norte'].copy()
sur   = df[df['dialecto'] == 'sur'].copy()

# cargar datos
print(f"Datos: {len(df)} tokens  (norte={len(norte)}, sur={len(sur)})\n")


# ANÁLISIS NORTE vs SUR — t-test + Cohen's d

print("=" * 60)
print("A) COMPARACION NORTE vs SUR — F1, F2, B1, B2")
print("=" * 60)

def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled

for col in ['F1_Hz', 'F2_Hz', 'B1_Hz', 'B2_Hz']:
    t, p = stats.ttest_ind(norte[col], sur[col], equal_var=False)
    d = cohens_d(norte[col], sur[col])
    sig = '*** significativo' if p < 0.05 else '(no significativo)'
    print(f"\n{col}:")
    print(f"  Norte: M={norte[col].mean():.1f}  SD={norte[col].std():.1f}")
    print(f"  Sur:   M={sur[col].mean():.1f}  SD={sur[col].std():.1f}")
    print(f"  Welch t={t:.3f}, p={p:.4f} {sig}  |  Cohen's d={d:.3f}")

# plot A: 4 subplots uno por variable
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
colores = {'norte': '#2166ac', 'sur': '#d6604d'}

for ax, col in zip(axes, ['F1_Hz', 'F2_Hz', 'B1_Hz', 'B2_Hz']):
    for dialecto, grupo in df.groupby('dialecto'):
        ax.hist(grupo[col], bins=25, alpha=0.55,
                color=colores[dialecto], label=dialecto.capitalize())
        # ejes
    ax.set_title(col.replace('_Hz', ''), fontsize=12)
    ax.set_xlabel('Hz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Distribuciones Norte vs Sur — F1, F2, B1, B2', fontsize=13)
plt.tight_layout()
plt.savefig('plot_distribuciones.png', dpi=150)
plt.close()
print("\n→ plot_distribuciones.png")


# k-means por separado para cada dialecto (solo F1/F2) — metodo del codo + silhouette
print("\n" + "=" * 60)
print("B) K-MEANS POR DIALECTO (solo F1/F2)")
print("=" * 60)

def codo_y_silhouette(datos, nombre, ax_codo, ax_sil, k_max=8):
    """
    Calcula inercia y silhouette para k=2..k_max
    Plotea el codo y el silhouette score
    Devuelve el k optimo segun silhouette
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(datos)

    inercias, silhouettes = [], []
    ks = range(2, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=20)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    k_optimo = ks[np.argmax(silhouettes)]

    ax_codo.plot(ks, inercias, 'o-', color=colores[nombre.lower()])
    ax_codo.set_title(f'{nombre} — codo', fontsize=11)
    ax_codo.set_xlabel('k')
    ax_codo.set_ylabel('Inercia')
    ax_codo.axvline(k_optimo, color='gray', linestyle='--', alpha=0.6)
    ax_codo.grid(True, alpha=0.3)

    ax_sil.plot(ks, silhouettes, 's-', color=colores[nombre.lower()])
    ax_sil.set_title(f'{nombre} — silhouette', fontsize=11)
    ax_sil.set_xlabel('k')
    ax_sil.set_ylabel('Silhouette score')
    ax_sil.axvline(k_optimo, color='gray', linestyle='--', alpha=0.6)
    ax_sil.grid(True, alpha=0.3)

    print(f"\n  {nombre}: k optimo = {k_optimo}  (silhouette={max(silhouettes):.3f})")
    for k, s in zip(ks, silhouettes):
        print(f"    k={k}: silhouette={s:.3f}")

    return k_optimo, scaler

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

k_norte, scaler_norte = codo_y_silhouette(
    norte[['F1_Hz', 'F2_Hz']], 'Norte', axes[0][0], axes[0][1])
k_sur, scaler_sur = codo_y_silhouette(
    sur[['F1_Hz', 'F2_Hz']], 'Sur', axes[1][0], axes[1][1])

fig.suptitle('Método del codo y Silhouette — Norte vs Sur', fontsize=13)
plt.tight_layout()
plt.savefig('plot_codo.png', dpi=150)
plt.close()
print("\n→ plot_codo.png")

# plot espacio vocalico con k optimo para cada dialecto
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colores_cluster = ['#e41a1c','#377eb8','#4daf4a','#ff7f00','#984ea3',
                   '#a65628','#f781bf','#999999']

for ax, (subdf, nombre, k_opt, scaler) in zip(axes, [
    (norte, 'Norte', k_norte, scaler_norte),
    (sur,   'Sur',   k_sur,   scaler_sur)
]):
    X = scaler.transform(subdf[['F1_Hz', 'F2_Hz']])
    km = KMeans(n_clusters=k_opt, random_state=SEMILLA, n_init=20)
    labels = km.fit_predict(X)

    for cl in range(k_opt):
        mask = labels == cl
        ax.scatter(subdf['F2_Hz'].values[mask],
                   subdf['F1_Hz'].values[mask],
                   c=colores_cluster[cl], alpha=0.45, s=30,
                   label=f'Cluster {cl+1}')

    # centroides en Hz reales
    cents_hz = scaler.inverse_transform(km.cluster_centers_)
    for i, (f2c, f1c) in enumerate(zip(cents_hz[:, 1], cents_hz[:, 0])):
        ax.scatter(f2c, f1c, c=colores_cluster[i],
                   s=200, marker='*', edgecolors='black', zorder=5)
        ax.annotate(f'C{i+1}', (f2c, f1c),
                    textcoords='offset points', xytext=(6, 4), fontsize=11)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)', fontsize=11)
    ax.set_ylabel('F1 (Hz)', fontsize=11)
    ax.set_title(f'{nombre} — k={k_opt} clusters', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Espacio vocálico por dialecto — k óptimo por silhouette', fontsize=13)
plt.tight_layout()
plt.savefig('plot_kmeans_dialectos.png', dpi=150)
plt.close()
print("→ plot_kmeans_dialectos.png")


# k-means con vs sin bandwidth (k=5, datos completos)
print("\n" + "=" * 60)
print("C) K-MEANS CON vs SIN BANDWIDTH (k=5, datos completos)")
print("=" * 60)

K_FIJO = 5

def kmeans_y_tabla(datos, columnas, nombre):
    scaler = StandardScaler()
    X = scaler.fit_transform(datos[columnas])
    km = KMeans(n_clusters=K_FIJO, random_state=SEMILLA, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    datos = datos.copy()
    datos['cluster'] = labels
    tabla = datos.groupby(['dialecto', 'cluster']).size().unstack(fill_value=0)
    print(f"\n  {nombre} — silhouette={sil:.3f}")
    print(tabla)
    return labels, sil

labels_2d, sil_2d = kmeans_y_tabla(df, ['F1_Hz', 'F2_Hz'], 'Solo F1/F2 (2D)')
labels_4d, sil_4d = kmeans_y_tabla(df, ['F1_Hz', 'F2_Hz', 'B1_Hz', 'B2_Hz'], 'F1/F2/B1/B2 (4D)')

print(f"\n  Mejora de silhouette al añadir bandwidth: {sil_4d - sil_2d:+.3f}")

# plot comparacion 2D vs 4D (proyectado en F1/F2 para visualizar)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
titulos = ['Sin bandwidth (F1/F2)', 'Con bandwidth (F1/F2/B1/B2)']

for ax, labels, titulo in zip(axes, [labels_2d, labels_4d], titulos):
    for cl in range(K_FIJO):
        mask = labels == cl
        ax.scatter(df['F2_Hz'].values[mask], df['F1_Hz'].values[mask],
                   c=colores_cluster[cl], alpha=0.4, s=25, label=f'C{cl+1}')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)')
    ax.set_ylabel('F1 (Hz)')
    ax.set_title(titulo, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Efecto del bandwidth en la estructura de clusters (k=5)', fontsize=13)
plt.tight_layout()
plt.savefig('plot_bandwidth_vs_no.png', dpi=150)
plt.close()
print("\n→ plot_bandwidth_vs_no.png")


# subgrupos dentro del sur (k-means con F1/F2/B1/B2, metodo del codo + silhouette)

print("\n" + "=" * 60)
print("D) SUBGRUPOS DENTRO DEL SUR")
print("=" * 60)

scaler_sur4d = StandardScaler()
X_sur = scaler_sur4d.fit_transform(sur[['F1_Hz', 'F2_Hz', 'B1_Hz', 'B2_Hz']])

inercias_sur, silhouettes_sur = [], []
ks = range(2, 8)
for k in ks:
    km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=20)
    labels = km.fit_predict(X_sur)
    inercias_sur.append(km.inertia_)
    silhouettes_sur.append(silhouette_score(X_sur, labels))

k_sur_4d = ks[np.argmax(silhouettes_sur)]
print(f"\n  k optimo dentro del sur (4D): {k_sur_4d}")
for k, s in zip(ks, silhouettes_sur):
    print(f"    k={k}: silhouette={s:.3f}")

# plot subgrupos del sur
km_sur = KMeans(n_clusters=k_sur_4d, random_state=SEMILLA, n_init=20)
labels_sur = km_sur.fit_predict(X_sur)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# espacio F1/F2
ax = axes[0]
for cl in range(k_sur_4d):
    mask = labels_sur == cl
    ax.scatter(sur['F2_Hz'].values[mask], sur['F1_Hz'].values[mask],
               c=colores_cluster[cl], alpha=0.5, s=40, label=f'Subgrupo {cl+1}')
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel('F2 (Hz)')
ax.set_ylabel('F1 (Hz)')
ax.set_title(f'Sur — {k_sur_4d} subgrupos en F1/F2', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# espacio B1/B2 (la dimension nueva)
ax = axes[1]
for cl in range(k_sur_4d):
    mask = labels_sur == cl
    ax.scatter(sur['B2_Hz'].values[mask], sur['B1_Hz'].values[mask],
               c=colores_cluster[cl], alpha=0.5, s=40, label=f'Subgrupo {cl+1}')
ax.set_xlabel('B2 (Hz)')
ax.set_ylabel('B1 (Hz)')
ax.set_title(f'Sur — {k_sur_4d} subgrupos en B1/B2', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# centroides en Hz reales
cents = scaler_sur4d.inverse_transform(km_sur.cluster_centers_)
print("\n  Centroides de subgrupos del sur:")
print(f"  {'Subgrupo':>10} {'F1':>8} {'F2':>8} {'B1':>8} {'B2':>8}")
for i, c in enumerate(cents):
    print(f"  {i+1:>10} {c[0]:>8.0f} {c[1]:>8.0f} {c[2]:>8.0f} {c[3]:>8.0f}")

fig.suptitle('Subgrupos dentro del Sur peninsular (F1/F2/B1/B2)', fontsize=13)
plt.tight_layout()
plt.savefig('plot_subgrupos_sur.png', dpi=150)
plt.close()
print("\n→ plot_subgrupos_sur.png")

print("\n Analisis completo.")
print("Plots generados: plot_distribuciones.png, plot_codo.png,")
print("                 plot_kmeans_dialectos.png, plot_bandwidth_vs_no.png,")
print("                 plot_subgrupos_sur.png")

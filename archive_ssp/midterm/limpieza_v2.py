import pandas as pd

df = pd.read_csv('results_formants.csv')

# Drop rows with no dialect label
df = df.dropna(subset=['dialecto'])

# Filtro F1/F2 — rangos plausibles para vocales del español
df = df[(df['F1_Hz'] >= 200) & (df['F1_Hz'] <= 900)]
df = df[(df['F2_Hz'] >= 700) & (df['F2_Hz'] <= 2800)]

# Filtro B1/B2 — bandwidths plausibles
# B muy pequeño (<50 Hz): artefacto numerico de Praat
# B muy grande (>600 Hz): frame no vocalico, ruido o consonante
df = df[(df['B1_Hz'] >= 30) & (df['B1_Hz'] <= 800)]
df = df[(df['B2_Hz'] >= 30) & (df['B2_Hz'] <= 900)]

print("Shape after cleaning:", df.shape)
print("\nDialect counts:")
print(df['dialecto'].value_counts())
print("\nStats after cleaning:")
print(df[['F1_Hz', 'F2_Hz', 'B1_Hz', 'B2_Hz']].describe())

df.to_csv('results_formants_clean.csv', index=False)
print("\nSaved to results_formants_clean.csv")

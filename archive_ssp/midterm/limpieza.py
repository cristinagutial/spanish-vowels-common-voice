import pandas as pd

df = pd.read_csv('results_formants.csv')

# Drop rows with no dialect label (orphan files from previous run)
df = df.dropna(subset=['dialecto'])

# Drop acoustic outliers — outside plausible vowel ranges for Spanish
df = df[(df['F1_Hz'] >= 200) & (df['F1_Hz'] <= 900)]
df = df[(df['F2_Hz'] >= 700) & (df['F2_Hz'] <= 2800)]

print("Shape after cleaning:", df.shape)
print("\nDialect counts:")
print(df['dialecto'].value_counts())
print("\nStats after cleaning:")
print(df[['F1_Hz', 'F2_Hz']].describe())

df.to_csv('results_formants_clean.csv', index=False)
print("\nSaved to results_formants_clean.csv")
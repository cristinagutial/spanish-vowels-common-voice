import pandas as pd
import os
import shutil

ruta_clips = r"C:\Users\x421\OneDrive\Desktop\CompLing\Speech_Project\Data\cv-corpus-25.0-2026-03-09\es\clips"
ruta_destino = "audios_filtrados"

def sacar_datos():
    """
    I take the audios out of the corpus and clean them to be able to process them
    """
    df = pd.read_csv('validated.tsv', sep='\t', low_memory=False)

    norte_label = 'España: Norte peninsular'
    sur_label   = 'España: Sur peninsular'

    df_norte = df[df['accents'].str.contains(norte_label, na=False)].sample(n=300)
    df_sur   = df[df['accents'].str.contains(sur_label,   na=False)].sample(n=300)

    df_unido = pd.concat([df_norte, df_sur])
    df_unido['dialecto'] = df_unido['accents'].apply(
        lambda x: 'norte' if norte_label in str(x) else 'sur'
    )
    df_unido[['path', 'dialecto']].to_csv('metadata_filtrada.csv', index=False)

    os.makedirs(ruta_destino, exist_ok=True)

    for archivo in df_unido['path']:
        ruta_origen = os.path.join(ruta_clips, archivo)
        ruta_final  = os.path.join(ruta_destino, archivo)
        try:
            shutil.copy2(ruta_origen, ruta_final)
            print(f"Copiado: {archivo}")
        except FileNotFoundError:
            print(f"Skipped (not found): {archivo}") 

sacar_datos()
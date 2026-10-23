import pandas as pd
import unicodedata
import re

"""THIS CODE DIDN'T WORK, SO I NEED A DIFFERENT APPROACH"""


# here i define the vowel ranges to filter out what i dont need
RANGOS_VALIDACION = {
    'a': {'F1': (500, 950), 'F2': (800, 2000)},
    'e': {'F1': (320, 700), 'F2': (1300, 2500)},
    'i': {'F1': (200, 500), 'F2': (1700, 2900)},
    'o': {'F1': (300, 680), 'F2': (600, 1500)},
    'u': {'F1': (200, 500), 'F2': (550, 1300)},
}


def acusticamente_plausible(vocal, f1, f2):
    """
    True if f1 and f2 are realistic for the vowel that was extracted.
    I used wide ranges on purpose because I want to eliminate wrong measurements
    
    """
    if vocal not in RANGOS_VALIDACION:
        return False
    r = RANGOS_VALIDACION[vocal]
    return (r['F1'][0] <= f1 <= r['F1'][1] and
            r['F2'][0] <= f2 <= r['F2'][1])


# logica d las silabas tonicas
VOCALES_STR    = 'aeiouáéíóúü'
VOCALES_TILDE  = set('áéíóú')
MAP_TILDE      = str.maketrans('áéíóúü', 'aeiouu')

# palabras que se tienen que ignorar
PALABRAS_FUNCION = {
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'de', 'del', 'al', 'a', 'en', 'con', 'por', 'para', 'sin',
    'que', 'se', 'le', 'lo', 'me', 'te', 'nos', 'les',
    'y', 'o', 'u', 'e', 'ni', 'pero', 'sino', 'porque', 'como',
    'es', 'son', 'era', 'fue', 'ser', 'han', 'ha', 'he',
    'su', 'sus', 'mi', 'mis', 'tu', 'tus',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa',
    'muy', 'mas', 'mas', 'ya', 'no', 'si', 'tambien',
}


def es_vocal(c):
    return c.lower() in VOCALES_STR


def tiene_tilde(c):
    return c in VOCALES_TILDE


def silabas_de(palabra):
    """
    I separate all the sylabels bc I assume that they will probably finish in a vowel.
    I think this is enough to identify the tonic vowel
    """
    silabas, actual = [], ''
    for c in palabra:
        actual += c
        if es_vocal(c.lower()):
            silabas.append(actual)
            actual = ''
    if actual and silabas:
        silabas[-1] += actual  # consonante final va a la ultima silaba
    elif actual:
        silabas.append(actual)
    return silabas


def vocal_tonica_de(palabra):
    """
    This gives me the tonic vowel of the word (normalized)
    I follow the accentuation rules in spanish to find it 
    (aguda, llana y esdrújula, i dont know how its called in english)
    """
    palabra_l = palabra.lower()

    if not any(es_vocal(c) for c in palabra_l):
        return None

    # if theres an accent, then this is a tonic syllable
    for c in palabra_l:
        if tiene_tilde(c):
            return c.translate(MAP_TILDE)

    silabas = silabas_de(palabra_l)
    if not silabas:
        return None

    ultima_letra = palabra_l[-1] if palabra_l else ''
    if ultima_letra in 'aeiouáéíóúns' and len(silabas) >= 2:
        silaba_tonica = silabas[-2]   # llana
    else:
        silaba_tonica = silabas[-1]   # aguda

    for c in silaba_tonica:
        if es_vocal(c):
            return c.translate(MAP_TILDE)
    return None


def normalizar_palabra(palabra):
    """
    remove accents and puts everything in lowercase
    """
    nfkd = unicodedata.normalize('NFD', palabra)
    sin_tildes = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return sin_tildes.lower()


def vocales_tonicas_frase(frase):
    """
    Gives me back all the tonic vowels
    """
    palabras = re.findall(r'[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+', frase)
    vocales = set()
    for p in palabras:
        p_norm = normalizar_palabra(p)
        if len(p_norm) <= 3 or p_norm in PALABRAS_FUNCION:
            continue
        v = vocal_tonica_de(p)
        if v:
            vocales.add(v)
    return vocales


def inferir_vocal_clip(frase):
    """
    Gives me the vowel tag for each clip or None if it isnt sure
    I took a conservative approach
    """
    vocales = vocales_tonicas_frase(frase)
    if len(vocales) == 1:
        return vocales.pop()
    return None


# main pipeline

def main():
    print("Cargando datos...")
    df_formants = pd.read_csv('results_formants_clean.csv')
    df_meta     = pd.read_csv('metadata_filtrada.csv')

    try:
        df_tsv = pd.read_csv(
            'validated.tsv', sep='\t',
            usecols=['path', 'sentence'],
            low_memory=False
        )
    except FileNotFoundError:
        print("\n validated.tsv not found")
        return

    # unir transcripciones
    df_meta = df_meta.merge(df_tsv, on='path', how='left')
    df = df_formants.merge(
        df_meta,
        left_on='nombre_archivo',
        right_on='path',
        how='left'
    )
    if 'dialecto_y' in df.columns:
        df.drop(columns=['dialecto_y'], inplace=True)
        df.rename(columns={'dialecto_x': 'dialecto'}, inplace=True)

    n_total = len(df)
    print(f"Filas de partida: {n_total}")

    # etiqueta textual 
    df['vocal'] = df['sentence'].apply(
        lambda s: inferir_vocal_clip(s) if isinstance(s, str) else None
    )

    n_ambiguos = df['vocal'].isna().sum()
    print(f"  Descartados por ambiguedad textual: {n_ambiguos}")

    df = df[df['vocal'].notna()].copy()

    # validacion acustica suave
    df['acust_ok'] = df.apply(
        lambda r: acusticamente_plausible(r['vocal'], r['F1_Hz'], r['F2_Hz']),
        axis=1
    )
    n_acust = (~df['acust_ok']).sum()
    print(f"  Descartados por validacion acustica: {n_acust}")
    df = df[df['acust_ok']].copy()

    # resultado
    print(f"\nFilas validas finales: {len(df)} / {n_total}")

    print("\nDistribucion por vocal:")
    print(df['vocal'].value_counts())

    print("\nDistribucion por dialecto x vocal:")
    tabla = df.groupby(['dialecto', 'vocal']).size().unstack(fill_value=0)
    print(tabla)

    # comprobar balance minimo (al menos 10 tokens por celda)
    print("Posible balance problem")
    problemas = [(d, v, tabla.loc[d, v])
                 for d in tabla.index for v in tabla.columns
                 if tabla.loc[d, v] < 10]
    if problemas:
        for d, v, n in problemas:
            print(f"  {d} / /{v}/: {n} tokens")
    else:
        print("Everything is ok!")

    columnas = ['nombre_archivo', 'F1_Hz', 'F2_Hz', 'dialecto', 'vocal', 'sentence']
    df[columnas].to_csv('results_formants_labeled.csv', index=False)
    print("Saved in results_formants_labeled.csv")


if __name__ == '__main__':
    main()
"""
Paths, seed and thresholds for the whole pipeline.
Everything is here so that no other script hard-codes a number.
"""

import os
from pathlib import Path

# misma semilla para todo lo que sea aleatorio (muestreo, bootstrap)
SEED = 42

ROOT = Path(__file__).resolve().parent.parent

# cada version tiene su carpeta de salida, para que el v3.1 no pise el v3.0:
#   SPEECH_OUTPUT=output_v31 ./speech/bin/python run_all.py --from 2
OUTPUT_NAME = os.environ.get("SPEECH_OUTPUT", "output")

# lo que ya esta en el disco
VALIDATED_TSV = ROOT / "validated.tsv"          # metadatos de Common Voice (corpus 26.0)
AUDIO_DIR     = ROOT / "audios_corpus"          # clips sacados del tar del corpus 26.0

# lo que produce el pipeline
OUTPUT      = ROOT / OUTPUT_NAME
DATA_DIR    = OUTPUT / "data"                   # tablas intermedias y finales
FIGURE_DIR  = OUTPUT / "figures"                # figuras
TABLE_DIR   = OUTPUT / "tables"                 # tablas de resultados
WORK_DIR    = OUTPUT / "work"                   # wavs, .lab, TextGrids

MFA_CORPUS_DIR    = WORK_DIR / "mfa_corpus"     # wav + .lab que le paso a MFA
MFA_ALIGNED_DIR   = WORK_DIR / "mfa_aligned"    # TextGrids que devuelve MFA

# ficheros intermedios con nombre, asi puedo relanzar un stage suelto
SAMPLE_CSV     = DATA_DIR / "sample_manifest.csv"    # s1: que clips y de que hablantes
TOKENS_RAW_CSV = DATA_DIR / "vowel_tokens_raw.csv"   # s4: una fila por vocal
TOKENS_CSV     = DATA_DIR / "vowel_tokens.csv"       # s5: ya normalizado y filtrado
TOKENS_SEXCEIL_CSV = DATA_DIR / "vowel_tokens_sexceil.csv"  # s5: lo mismo con el techo por sexo
SIBILANTS_CSV  = DATA_DIR / "sibilant_tokens.csv"    # s8: una fila por /s/

for _d in (OUTPUT, DATA_DIR, FIGURE_DIR, TABLE_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Common Voice mete la lista entera de regiones en el campo `accents`,
# asi que busco el prefijo y no la cadena completa
DIALECTS = {
    "norte":  "España: Norte peninsular",
    "sur":    "España: Sur peninsular",
    "mexico": "México",                          # grupo de control
}

# los dos dialectos del estudio
CORE_DIALECTS = ["norte", "sur"]

# Mexico va de control: si el metodo separa el español de America del peninsular
# pero no Norte de Sur, entonces el resultado nulo es de los dialectos y no del metodo
OUTGROUP_DIALECTS = ["mexico"]
ALL_DIALECTS = CORE_DIALECTS + OUTGROUP_DIALECTS

# tope de clips por hablante del v3.0, para que nadie domine una celda. El v3.1
# usa 40 (s1_sample --cap 40); los clips con rango > 8 son los que el v3.0 no midio
MAX_CLIPS_PER_SPEAKER = 8

MIN_CLIPS_PER_SPEAKER = 1

# tope de clips por dialecto, despues del tope por hablante. None = sin tope
MAX_CLIPS_PER_DIALECT = None

# tope de hablantes por dialecto. Sur solo tiene 188 en el corpus 26.0 y eso es
# lo que limita todo el estudio, asi que Norte y Sur van enteros. Mexico tiene
# 1526 disponibles, que son muchos mas de los que hacen falta para el control
MAX_SPEAKERS_PER_DIALECT = {"mexico": 300}

MFA_ENV        = "mfa"              # nombre del entorno de conda
MFA_ACOUSTIC   = "spanish_mfa"
MFA_DICTIONARY = "spanish_mfa"
MFA_SAMPLE_RATE = 16000             # MFA quiere 16 kHz mono
MFA_NUM_JOBS   = 8              # tengo 10 nucleos, dejo margen para el sistema

# las cinco vocales tal y como las etiqueta el modelo spanish_mfa.
# Las semiconsonantes (j, w) fuera: no son nucleo de silaba y sus formantes son transiciones
VOWELS = ["a", "e", "i", "o", "u"]

# techo de Burg por hablante: pruebo todos estos y me quedo con el que da menos
# varianza dentro de cada vocal (la idea de Escudero et al. 2009). Empiezo en
# 4000 y no en 4500 porque el 4500 estaba pensado para voces de mujer, y en la
# prueba con 6 hablantes un hombre ya caia en el borde
CEILING_GRID = [float(c) for c in range(4000, 6501, 100)]

# para elegir el techo, una vocal solo cuenta si el hablante tiene estos tokens,
# y hacen falta al menos tantas vocales distintas
MIN_TOKENS_FOR_CEILING = 3
MIN_VOWELS_FOR_CEILING = 3

# si un hablante no tiene datos para elegir, uso el techo por sexo de antes
CEILING_MALE   = 5000.0
CEILING_FEMALE = 5500.0
CEILING_UNKNOWN = 5250.0            # punto medio cuando Common Voice no tiene el dato

MAX_N_FORMANTS = 5.0

# vocales mas cortas que esto son solo transiciones a las consonantes de al lado
MIN_VOWEL_DURATION = 0.030          # segundos

# en los TextGrids los silencios son intervalos vacios. Al quitarlos queda un
# hueco entre dos fonemas, y cualquier hueco es una pausa
PAUSE_GAP = 0.001                   # segundos

# rangos plausibles para vocales del español, con los dos sexos dentro.
# Son anchos a proposito: estan para quitar fallos del tracker (un F1 de 3 kHz es
# un error de octava, no una vocal), no para imponer lo que espero encontrar
F1_RANGE = (150.0, 1100.0)
F2_RANGE = (500.0, 3200.0)

# outliers por hablante y vocal, en z
OUTLIER_Z = 3.0

# Lobanov con las medias de las cinco vocales: hacen falta las cinco, cada una
# con al menos estos tokens
MIN_TOKENS_PER_VOWEL_LOBANOV = 2

# un hablante cuyo triangulo vocalico no existe es un fallo del tracker, no un
# acento: por ejemplo todas las vocales con F2 = 1000 Hz. Con las medias en Hz,
# la /i/ tiene que estar por delante de la /u/ y la /a/ por debajo de las dos.
# Umbral 0: solo quito lo imposible. Con 500/100 Hz cortaba dentro de la
# distribucion real, sobre las mismas F2 que mide H1, y mas en Norte que en Sur
MIN_F2_I_MINUS_U = 0.0              # Hz
MIN_F1_A_MINUS_CLOSE = 0.0          # Hz, F1 /a/ menos el mayor de F1 /i/ y F1 /u/

# palabras gramaticales. En H1 y H2 las quito porque son atonas o se reducen,
# y antes de una /s/ final son casi todo lo que hay (los, las, es, sus, estos...)
FUNCTION_WORDS = {
    # articulos y contracciones
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "al", "del",
    # preposiciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante", "en", "entre",
    "hacia", "hasta", "mediante", "para", "por", "según", "sin", "so", "sobre", "tras",
    # conjunciones y relativos
    "y", "e", "o", "u", "ni", "que", "si", "pero", "sino", "porque", "aunque",
    "como", "cuando", "donde", "mientras", "pues", "cual", "cuales", "quien", "quienes",
    "cuyo", "cuya", "cuyos", "cuyas",
    # pronombres
    "me", "te", "se", "nos", "os", "le", "les", "yo", "tú", "él", "ella", "ello",
    "ellos", "ellas", "nosotros", "nosotras", "vosotros", "vosotras", "usted", "ustedes",
    "esto", "eso", "aquello",
    # posesivos
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras",
    # demostrativos y cuantificadores
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    "todo", "toda", "todos", "todas", "otro", "otra", "otros", "otras",
    "alguno", "alguna", "algunos", "algunas", "ninguno", "ninguna",
    "mucho", "mucha", "muchos", "muchas", "poco", "poca", "pocos", "pocas",
    "varios", "varias", "ambos", "ambas", "cada", "tanto", "tanta", "tantos", "tantas",
    "dos", "tres", "seis",
    # adverbios gramaticales
    "no", "ya", "muy", "más", "mas", "menos", "tan", "también", "tampoco",
    "entonces", "antes", "después", "así",
    # formas de ser, estar y haber
    "es", "son", "era", "eran", "fue", "fueron", "ser", "sido", "está", "están",
    "estaba", "estaban", "ha", "han", "he", "has", "hay", "había", "habían",
}

# H1: una celda hablante x vocal necesita al menos estos tokens tonicos
MIN_TOKENS_PER_CELL = 2

# H1: equivalencia. Una diferencia menor que esta d no cuenta como diferencia de
# dialecto. 0.30 es lo que median las diferencias Peninsular/Mexico mas pequeñas
# que el v2.0 si detectaba
TOST_BOUND = 0.30

# H2: vocales finales atonas antes de /s/ final (casas) y sin /s/ (casa)
H2_VOWELS = ["a", "e", "o"]
H2_PRIMARY_VOWELS = ["e", "o"]      # la prediccion fuerte es para las medias
MIN_TOKENS_PER_H2_CELL = 2

# H2a: la /s/. Mido en el 50% central del segmento, en el mp3 original (48 kHz),
# con un filtro de 1 a 11 kHz para quitar la sonoridad y el ruido de abajo
MIN_S_DURATION = 0.020
S_WINDOW = (0.25, 0.75)
S_BAND = (1000.0, 11000.0)
MIN_S_TOKENS_PER_CELL = 3

ALPHA = 0.05
BOOTSTRAP_N = 2000                  # remuestreos para los intervalos de confianza

DPI = 200

COLORS = {
    "norte":  "#2166ac",
    "sur":    "#d6604d",
    "mexico": "#4daf4a",
}

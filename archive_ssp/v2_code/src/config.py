"""
Paths, seed and thresholds for the whole pipeline.
Everything is here so that no other script hard-codes a number.
"""

from pathlib import Path

# misma semilla para todo lo que sea aleatorio (muestreo, k-means, bootstrap, CV)
SEED = 42

ROOT = Path(__file__).resolve().parent.parent

# lo que ya esta en el disco
VALIDATED_TSV = ROOT / "validated.tsv"          # metadatos de Common Voice (corpus 26.0)
AUDIO_DIR     = ROOT / "audios_corpus"          # clips sacados del tar del corpus 26.0

# la muestra de 903 clips del midterm, la guardo porque el stage 9 la necesita
LEGACY_AUDIO_DIR = ROOT / "audios_filtrados"

# lo que produce el pipeline nuevo
OUTPUT      = ROOT / "output"
DATA_DIR    = OUTPUT / "data"                   # tablas intermedias y finales
FIGURE_DIR  = OUTPUT / "figures"                # figuras del informe
TABLE_DIR   = OUTPUT / "tables"                 # tablas del informe
WORK_DIR    = OUTPUT / "work"                   # wavs, .lab, TextGrids

MFA_CORPUS_DIR    = WORK_DIR / "mfa_corpus"     # wav + .lab que le paso a MFA
MFA_ALIGNED_DIR   = WORK_DIR / "mfa_aligned"    # TextGrids que devuelve MFA

# ficheros intermedios con nombre, asi puedo relanzar un stage suelto
SAMPLE_CSV     = DATA_DIR / "sample_manifest.csv"    # s1: que clips y de que hablantes
TOKENS_RAW_CSV = DATA_DIR / "vowel_tokens_raw.csv"   # s4: una fila por vocal
TOKENS_CSV     = DATA_DIR / "vowel_tokens.csv"       # s5: ya normalizado y filtrado

# resultado del analisis del midterm, el stage 9 lo reproduce
LEGACY_CLEAN_CSV = ROOT / "results_formants_clean.csv"

for _d in (OUTPUT, DATA_DIR, FIGURE_DIR, TABLE_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Common Voice mete la lista entera de regiones en el campo `accents`,
# asi que busco el prefijo y no la cadena completa
DIALECTS = {
    "norte":  "España: Norte peninsular",
    "sur":    "España: Sur peninsular",
    "mexico": "México",                          # grupo externo, necesita el corpus entero
}

# los dos dialectos del estudio
CORE_DIALECTS = ["norte", "sur"]

# Mexico va de control: si el metodo separa el español de America del peninsular
# pero no Norte de Sur, entonces el resultado nulo es de los dialectos y no del metodo
OUTGROUP_DIALECTS = ["mexico"]
ALL_DIALECTS = CORE_DIALECTS + OUTGROUP_DIALECTS

# tope de clips por hablante. En el midterm un solo hablante tenia el 80% de la
# celda del Sur y de ahi salian los efectos falsos, asi que aqui lo corto
MAX_CLIPS_PER_SPEAKER = 8

# un hablante necesita unos cuantos clips para que Lobanov tenga sentido
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

# El algoritmo Burg de Praat necesita un techo de frecuencia acorde al tracto vocal.
# Poner 5000 Hz para todo el mundo (como hacia en el midterm) estima mal las voces
# femeninas, y aqui las celdas de dialecto no estan equilibradas por sexo
CEILING_MALE   = 5000.0
CEILING_FEMALE = 5500.0
CEILING_UNKNOWN = 5250.0            # punto medio cuando Common Voice no tiene el dato

MAX_N_FORMANTS = 5.0

# vocales mas cortas que esto son solo transiciones a las consonantes de al lado
MIN_VOWEL_DURATION = 0.030          # segundos

# mido los formantes en estos puntos de la vocal. El del medio es el valor de
# siempre; los del 25% y 75% sirven para ver si el formante se mueve
MEASURE_POINTS = [0.25, 0.50, 0.75]

# rangos plausibles para vocales del español, con los dos sexos dentro.
# Son anchos a proposito: estan para quitar fallos del tracker (un F1 de 3 kHz es
# un error de octava, no una vocal), no para imponer lo que espero encontrar
F1_RANGE = (150.0, 1100.0)
F2_RANGE = (500.0, 3200.0)
B1_RANGE = (10.0, 900.0)
B2_RANGE = (10.0, 1000.0)

# outliers por hablante y vocal, en z. Ahora puedo hacerlo porque se que vocal es
OUTLIER_Z = 3.0

# por debajo de este SNR marco el clip. Los bandwidths se estropean con ruido,
# que es la otra explicacion posible del B2 del midterm
MIN_SNR_DB = 5.0

ALPHA = 0.05
BOOTSTRAP_N = 2000                  # remuestreos para los intervalos de confianza
N_CV_FOLDS = 5

DPI = 200

COLORS = {
    "norte":  "#2166ac",
    "sur":    "#d6604d",
    "mexico": "#4daf4a",
}

VOWEL_COLORS = {
    "a": "#e41a1c",
    "e": "#377eb8",
    "i": "#4daf4a",
    "o": "#ff7f00",
    "u": "#984ea3",
}

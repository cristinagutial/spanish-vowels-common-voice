import parselmouth
import numpy as np
import os
import pandas as pd


def extraer_f1_f2(ruta_audio):
    """
    Loads an audio file and extracts F1 and F2 formants using Praat's
    Burg algorithm via parselmouth.

    Instead of a single peak frame (which could be noise or a consonant),
    we take a 50ms window around the loudest moment and compute the median.
    The median is robust against NaN values that Praat returns for unvoiced frames.
    """
    sonido = parselmouth.Sound(ruta_audio)

    # buscar el nucleo d la silaba
    intensidad = sonido.to_intensity()
    valores_intensidad = intensidad.values[0]
    indice_maximo = np.argmax(valores_intensidad)
    tiempo_pico = intensidad.get_time_from_frame_number(indice_maximo + 1)

    # hacer una ventana alrededor del pico, clippear los limites del audio
    ventana = 0.025  # 25ms a cada lado (50 ms en total)
    t_inicio = max(0, tiempo_pico - ventana)
    t_fin    = min(sonido.duration, tiempo_pico + ventana)

    # sampleo de 10 puntos en cada ventana
    tiempos = np.linspace(t_inicio, t_fin, 10)

    # 5000 Hz ceiling: better calibrated for mixed-gender Spanish speakers
    formantes = sonido.to_formant_burg(max_number_of_formants=5, maximum_formant=5000)

    f1_vals = [formantes.get_value_at_time(1, t) for t in tiempos]
    f2_vals = [formantes.get_value_at_time(2, t) for t in tiempos]

    # esto ayuda a ignorar los unvoiced segments
    f1 = float(np.nanmedian(f1_vals))
    f2 = float(np.nanmedian(f2_vals))

    return f1, f2


def procesar_carpeta():
    """
    Iterates over all .mp3 files in audios_filtrados, extracts F1/F2,
    then joins with dialect metadata and saves a single CSV.
    """
    carpeta_audios = "audios_filtrados"
    resultados_foneticos = []
    print("Extracting formants...")

    for archivo in os.listdir(carpeta_audios):
        if archivo.endswith('.mp3'):
            ruta_completa = os.path.join(carpeta_audios, archivo)
            try:
                f1, f2 = extraer_f1_f2(ruta_completa)
                resultados_foneticos.append({
                    'nombre_archivo': archivo,
                    'F1_Hz': round(f1, 1),
                    'F2_Hz': round(f2, 1)
                })
                print(f"{archivo} | F1: {f1:.1f} Hz | F2: {f2:.1f} Hz")
            except Exception as e:
                print(f"ERROR with {archivo}: {e}")

    # hacer tabla
    df_resultados = pd.DataFrame(resultados_foneticos)

    # juntar labels d dialectos con los del preprocesador
    df_meta = pd.read_csv('metadata_filtrada.csv')
    df_resultados = df_resultados.merge(
        df_meta,
        left_on='nombre_archivo',
        right_on='path',
        how='left'
    )

    df_resultados.drop(columns=['path'], inplace=True)

    df_resultados.to_csv("results_formants.csv", index=False)
    print(f"\nDone. {len(df_resultados)} files processed → results_formants.csv")


procesar_carpeta()


# test con mi voz
archivo_de_prueba = "prueba.wav"  # CHANGE THIS

try:
    f1, f2 = extraer_f1_f2(archivo_de_prueba)
    print(f"Test | F1 ≈ {f1:.2f} Hz | F2 ≈ {f2:.2f} Hz")
except Exception as e:
    print(f"Test error: {e}")




# test


# here im using a recording of myself to check that it works
archivo_de_prueba = "prueba.wav"  # CHANGE THIS

try:
    f1, f2 = extraer_f1_f2(archivo_de_prueba)
    print(f"F1 is aprox: {f1:.2f} Hz")
    print(f"F2 is aprox: {f2:.2f} Hz")
except Exception as e:
    print(f"Error: {e}")
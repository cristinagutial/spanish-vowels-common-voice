import parselmouth
import numpy as np
import os
import pandas as pd


def extraer_formantes(ruta_audio):
    """
    Loads an audio file and extracts F1, F2 and their bandwidths B1, B2
    using Praat's Burg algorithm via parselmouth.

    Instead of a single peak frame (which could be noise or a consonant),
    we take a 50ms window around the loudest moment and compute the median.
    The median is robust against NaN values that Praat returns for unvoiced frames.
    """
    sonido = parselmouth.Sound(ruta_audio)

    # buscar el nucleo de la silaba
    intensidad = sonido.to_intensity()
    valores_intensidad = intensidad.values[0]
    indice_maximo = np.argmax(valores_intensidad)
    tiempo_pico = intensidad.get_time_from_frame_number(indice_maximo + 1)

    # ventana de 50ms alrededor del pico
    ventana = 0.025  # 25ms a cada lado
    t_inicio = max(0, tiempo_pico - ventana)
    t_fin    = min(sonido.duration, tiempo_pico + ventana)

    # 10 puntos de muestreo dentro de la ventana
    tiempos = np.linspace(t_inicio, t_fin, 10)

    # 5000 Hz ceiling: calibrado para hablantes mixtos de español
    formantes = sonido.to_formant_burg(max_number_of_formants=5, maximum_formant=5000)

    # Formantes F1 y F2
    f1_vals = [formantes.get_value_at_time(1, t) for t in tiempos]
    f2_vals = [formantes.get_value_at_time(2, t) for t in tiempos]

    # Bandwidths B1 y B2
    # get_bandwidth_at_time() devuelve NaN en frames no sonoros, igual que los formantes
    b1_vals = [formantes.get_bandwidth_at_time(1, t) for t in tiempos]
    b2_vals = [formantes.get_bandwidth_at_time(2, t) for t in tiempos]

    f1 = float(np.nanmedian(f1_vals))
    f2 = float(np.nanmedian(f2_vals))
    b1 = float(np.nanmedian(b1_vals))
    b2 = float(np.nanmedian(b2_vals))

    return f1, f2, b1, b2


def procesar_carpeta():
    """
    Iterates over all .mp3 files in audios_filtrados, extracts F1/F2/B1/B2,
    then joins with dialect metadata and saves a single CSV.
    """
    carpeta_audios = "audios_filtrados"
    resultados_foneticos = []
    print("Extracting formants and bandwidths...")

    for archivo in os.listdir(carpeta_audios):
        if archivo.endswith('.mp3'):
            ruta_completa = os.path.join(carpeta_audios, archivo)
            try:
                f1, f2, b1, b2 = extraer_formantes(ruta_completa)
                resultados_foneticos.append({
                    'nombre_archivo': archivo,
                    'F1_Hz': round(f1, 1),
                    'F2_Hz': round(f2, 1),
                    'B1_Hz': round(b1, 1),
                    'B2_Hz': round(b2, 1),
                })
                print(f"{archivo} | F1:{f1:.0f} F2:{f2:.0f} | B1:{b1:.0f} B2:{b2:.0f}")
            except Exception as e:
                print(f"ERROR with {archivo}: {e}")

    df_resultados = pd.DataFrame(resultados_foneticos)

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


# test con mi voz (CAMBIAR)
archivo_de_prueba = "prueba.wav"

try:
    f1, f2, b1, b2 = extraer_formantes(archivo_de_prueba)
    print(f"Test | F1:{f1:.1f} Hz  F2:{f2:.1f} Hz | B1:{b1:.1f} Hz  B2:{b2:.1f} Hz")
except Exception as e:
    print(f"Test error: {e}")

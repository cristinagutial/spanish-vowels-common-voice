# Abre cada vocal de la muestra en Praat, una detras de otra, para medirla a mano.
#
# Praat > Open Praat script... > este fichero > Run.
# En cada vocal: Formants > Show formants (si no se ven), pon el cursor en el
# centro de la vocal, Formants > Get first formant / Get second formant, y apunta
# los valores en track_check_sample.csv (F1_praat, F2_praat, cristina_check).

table = Read Table from tab-separated file: "track_check_sample.tsv"
n = Get number of rows

for i to n
    selectObject: table
    id = Get value: i, "token_id"
    wav$ = Get value: i, "wav"
    grid$ = Get value: i, "textgrid"
    start = Get value: i, "start"
    end = Get value: i, "end"
    ceiling = Get value: i, "ceiling_Hz"

    sound = Read from file: wav$
    textgrid = Read from file: grid$
    selectObject: sound, textgrid
    View & Edit
    editor: textgrid
        Formant settings: ceiling, 5, 0.025, 30, 1
        Zoom: start - 0.05, end + 0.05
        Move cursor to: (start + end) / 2
    endeditor

    beginPause: "Token " + string$(id) + " of " + string$(n)
        comment: "Vowel from " + fixed$(start, 3) + " to " + fixed$(end, 3) + " s, ceiling " + string$(ceiling) + " Hz."
        comment: "Measure F1 and F2 at the cursor and write them in track_check_sample.csv."
    clicked = endPause: "Next", "Stop", 1

    removeObject: sound, textgrid
    if clicked = 2
        removeObject: table
        exitScript: "Stopped at token ", id
    endif
endfor

removeObject: table

#!/usr/bin/env python
"""
Run the whole study.

    ./speech/bin/python run_all.py              # every stage
    ./speech/bin/python run_all.py --from 4     # continue from the extraction
    ./speech/bin/python run_all.py --only 9     # just the speaker audit
    ./speech/bin/python run_all.py --list       # list the stages

Stages 2 to 4 are the slow ones. Each stage saves its output, so the later ones
can be re-run on their own.
"""

import argparse
import sys
import time
import traceback

STAGES = [
    (1, "s1_sample",    "Speaker-aware sampling"),
    (2, "s2_prepare",   "Audio conversion and transcripts"),
    (3, "s3_align",     "Forced alignment (MFA)"),
    (4, "s4_extract",   "Per-vowel formant extraction"),
    (5, "s5_normalize", "Quality control and normalisation"),
    (6, "s6_stats",     "Corrected statistical analysis"),
    (7, "s7_cluster",   "Clustering against vowel labels"),
    (8, "s8_classify",  "Classification benchmarks"),
    (9, "s9_pseudorep", "Speaker audit of the midterm analysis"),
    (10, "s10_codas",   "Andalusian coda-/s/ hypothesis"),
    (11, "plots",       "Report figures"),
]


def run_stage(number, module_name, description):
    import importlib

    print(f"\n{'#' * 74}")
    print(f"# STAGE {number}: {description}")
    print(f"{'#' * 74}\n")

    started = time.time()
    try:
        module = importlib.import_module(f"src.{module_name}")
        module.main()
    except SystemExit as exc:
        if exc.code:
            print(f"\nStage {number} exited with status {exc.code}")
            return False
    except Exception:
        traceback.print_exc()
        print(f"\nStage {number} failed.")
        return False

    print(f"\n[stage {number} finished in {time.time() - started:.1f}s]")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="first stage to run")
    parser.add_argument("--to", dest="end", type=int, default=99,
                        help="last stage to run")
    parser.add_argument("--only", type=int, nargs="+", help="run only these stages")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    if args.list:
        for number, module_name, description in STAGES:
            print(f"  {number:>2}  {module_name:<14} {description}")
        return 0

    if args.only:
        selected = [s for s in STAGES if s[0] in args.only]
    else:
        selected = [s for s in STAGES if args.start <= s[0] <= args.end]

    if not selected:
        print("No stages selected.")
        return 1

    print(f"Running {len(selected)} stage(s): {[s[0] for s in selected]}")
    started = time.time()

    for number, module_name, description in selected:
        if not run_stage(number, module_name, description):
            print(f"\nStopped at stage {number}.")
            return 1

    print(f"\n{'#' * 74}")
    print(f"# All stages completed in {time.time() - started:.1f}s")
    print(f"# Tables  -> output/tables/")
    print(f"# Figures -> output/figures/")
    print(f"{'#' * 74}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

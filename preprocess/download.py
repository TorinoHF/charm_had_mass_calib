"""
Script to download unmerged outputs from hyperloop
"""

import os
import sys
import argparse
import math
import multiprocessing
import uproot
import glob

def get_files_from_directory(input_directory, run, is_slim, load_merged_runs):
    """
    Function to get all the files from a specific directory
    """


    if is_slim or load_merged_runs:
        os.system(f"alien_find alien://{input_directory} "
                  f"AnalysisResults.root -r > outputs_{run}.txt")
    else:
        os.system(f"alien_find alien://{input_directory} "
                  f"AOD/*/AnalysisResults.root > outputs_{run}.txt")
    print(f"Looked for files in alien://{input_directory} and saved the list in outputs_{run}.txt")
    lines = []
    check_unmerged = False
    with open(f"outputs_{run}.txt") as file:
        lines = [(line.rstrip(), run) for line in file]

        if len(lines) == 0:
            check_unmerged = True

    if check_unmerged:
        os.system(f"alien_find alien://{input_directory} "
                  f"*/AnalysisResults.root > outputs_{run}.txt")

        with open(f"outputs_{run}.txt") as file:
            for line in file:
                if "Stage" in line: # we do not want partially merged files
                    continue
                lines.append((line.rstrip(), run))

    if not os.path.isdir(f"run_{run}"):
        os.mkdir(f"run_{run}")
    return lines


def download_file(task_id, file, run, runs, an_res_only):
    """
    Function to be executed in parallel that downloads all the files for a specific run
    """

    os.system(f"alien_cp {file} file:run_{run}/AnalysisResults_{task_id:03d}.root")
    # Only select files which have the wanted run numbers
    if runs:
        with uproot.open(f"run_{run}/AnalysisResults_{task_id:03d}.root") as f:
            run_numbers = f["eventselection-run3"]["luminosity"]["hCounterTVX"].axis().labels()
            run_numbers = [int(run) for run in run_numbers if run.isdigit()]
            # check if any of the run numbers is in the list
            if not any(run in runs for run in run_numbers):
                os.system(f"rm run_{run}/AnalysisResults_{task_id:03d}.root")
                return

    if not an_res_only:
        if os.system(f"alien_cp {file.replace('AnalysisResults', 'AO2D')} file:run_{run}/AO2D_{task_id:03d}.root") != 0:
            # AO2D not found, let's delete also AnalysisResults.root
            os.system(f"rm run_{run}/AnalysisResults_{task_id:03d}.root")

    print(f"Processing task {task_id} - DONE")


def download_and_merge(input_file, num_workers, suffix, n_merged_files, is_slim, load_merged_runs, runs, an_res_only):
    """
    Main function to download and merge output files from hyperloop
    """

    output_directories = []
    with open(input_file, 'r') as file:
        for line in file:
            # Split line by comma or any delimiter
            parts = line.strip().split(',')
            for part in parts:
                output_directories.append(part)

    num_workers = min(num_workers, os.cpu_count())  # Get the number of available CPU cores)

    # Get the runs from the runs file
    if runs is not None:
        with open(runs, 'r') as file:
            runs = file.read().strip()
        runs = runs.split(',')
        runs = [run.strip() for run in runs if run.strip().isdigit()]
        runs = [int(run) for run in runs]

    with multiprocessing.Pool(processes=num_workers) as pool:
        files = pool.starmap(
            get_files_from_directory,
            [(directory, run, is_slim, load_merged_runs) for directory, run in zip(output_directories, runs)]
        )
    files = [item for sublist in files for item in sublist]  # Flatten the list of lists

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            download_file,
            [(i, file[0], file[1], runs, an_res_only) for i, file in enumerate(files)]
        )

    files_to_merge = []
    for train_dir in os.listdir("."):
        if os.path.isdir(train_dir) and "run_" in train_dir:
            for file in os.listdir(train_dir):
                if "AO2D" in file:
                    file_path = os.path.join(train_dir, file)
                    files_to_merge.append(file_path)

    n_files_perbunch = math.ceil(len(files_to_merge) / n_merged_files)
    nbunch = 0
    for ifile, file in enumerate(files_to_merge):
        if ifile % n_files_perbunch == 0:
            nbunch += 1
        os.system(f"echo {file} >> files_to_merge_{nbunch}.txt")

    if nbunch > 1:
        for bunch in range(nbunch):
            os.system(f"o2-aod-merger --input files_to_merge_{bunch}.txt --output "
                      f"AO2D{suffix}_{bunch}.root --max-size 100000000 --skip-parent-files-list")
    else:
        os.system(f"o2-aod-merger --input files_to_merge_1.txt --output "
                  f"AO2D{suffix}.root --max-size 100000000 --skip-parent-files-list")

    files = glob.glob("./run_*/AnalysisResults*.root")
    try:
        os.system(f"hadd -f AnalysisResults{suffix}.root {' '.join(files)}")
        os.system(f"rm -r run_*")
        os.system(f"rm outputs_run_*")
        os.system(f"rm files_to_merge_*")
    except Exception as e:
        print(f"Error occurred while merging files: {e}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Arguments")
    PARSER.add_argument("--input_file", "-i", metavar="text",
                        help="text input file with directories", required=True)
    PARSER.add_argument("--jobs", "-j", type=int, default=20,
                        help="number of workers", required=False)
    PARSER.add_argument("--suffix", "-s", metavar="text",
                        default="_LHC24_pass1_skimmed",
                        help="output file", required=False)
    PARSER.add_argument("--n_merged_files", "-n", type=int,
                        default=1, help="number of output merged AO2D files",
                        required=False)
    PARSER.add_argument("--slim", action="store_true", default=False,
                        help="option for slim outputs on hyperloop", required=False)
    PARSER.add_argument("--load_merged_runs", action="store_true", default=False,
                        help="option for loading merged runs", required=False)
    PARSER.add_argument("--runs", metavar="text", required=False, default=None,
                        help="text input file with run numbers to download (only for non-slim files)")
    PARSER.add_argument("--an_res_only", action="store_true", default=False,
                        help="option for only downloading analysis results (not AO2D)", required=False)
    ARGS = PARSER.parse_args()

    download_and_merge(ARGS.input_file, ARGS.jobs, ARGS.suffix, ARGS.n_merged_files, ARGS.slim, ARGS.load_merged_runs, ARGS.runs, ARGS.an_res_only)

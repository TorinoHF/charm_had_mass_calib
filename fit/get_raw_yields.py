"""Script to perform fitting of invariant mass distributions for Ds and D+ mesons.

Inputs are the ``MassDistributions.root`` files produced by preprocessing,
discovered under ``inputs.data_dir`` with layout
``<data_dir>/cent_<MIN>_<MAX>/<YEAR>/MassDistributions.root``. Each file holds
TH2 histograms ``h2D_<hadron>_Occ_<MIN>_<MAX>_Pt_<MIN>_<MAX>`` (mass on X,
score on Y). For every file they are summed across occupancy bins, projected
onto the mass (X) axis and written as ``h_mass_<PT_MIN*10>_<PT_MAX*10>_cent_<CMIN>_<CMAX>``
into a sibling ``projections.root`` that the fitter consumes.
"""
import argparse
import copy
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
from typing import Dict, List, Tuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../utils/fitter')))
from tqdm import tqdm
import uproot
import yaml
import pandas as pd
import numpy as np
import ROOT
from pypdf import PdfWriter
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(20)
from fit_handler import (
    FitHandler,
    BRInfo,
    CorrelatedBackground,
    CorrelatedBackgroundConfig,
    FitConfig
)
from hist_handler import HistHandler

# pylint: disable=no-member  # (ROOT dynamic members)

H2D_NAME_RE = re.compile(
    r"^h2D_[^_]+_Occ_(?P<occ_min>[\d.]+)_(?P<occ_max>[\d.]+)"
    r"_Pt_(?P<pt_min>[\d.]+)_(?P<pt_max>[\d.]+)$"
)
CENT_DIR_RE = re.compile(r"^cent_(?P<cmin>\d+)_(?P<cmax>\d+)$")


def fitconfig_to_dict(cfg: FitConfig) -> dict:
    """Convert FitConfig dataclass to dictionary with modified keys."""
    base = dataclasses.asdict(cfg)
    return {f"{k}_cfg": v for k, v in base.items()}


def discover_input_files(base_dir: str) -> Dict[str, List[Dict]]:
    """Find every ``MassDistributions.root`` under ``base_dir/cent_*/<year>/``.

    Intermediate ``jobs/`` outputs are ignored. Files are grouped by year so a
    single per-year pipeline can run a centrality-integrated fit followed by
    the per-centrality fits.
    """
    base_dir = os.path.expanduser(base_dir)
    by_year: Dict[str, List[Dict]] = {}
    for cent_dir in sorted(glob.glob(os.path.join(base_dir, "cent_*"))):
        m_cent = CENT_DIR_RE.match(os.path.basename(cent_dir))
        if not m_cent:
            continue
        cent_min = int(m_cent.group("cmin"))
        cent_max = int(m_cent.group("cmax"))
        for year_name in sorted(os.listdir(cent_dir)):
            mass_file = os.path.join(cent_dir, year_name, "MassDistributions.root")
            if not os.path.isfile(mass_file):
                continue
            by_year.setdefault(year_name, []).append({
                "input_path": mass_file,
                "cent_min": cent_min,
                "cent_max": cent_max,
            })
    return by_year


def _load_h2_per_pt(input_path: str) -> Dict[Tuple[float, float], "ROOT.TH2"]:
    """Read TH2s from one MassDistributions.root, summing across occupancy bins."""
    f_in = ROOT.TFile.Open(input_path, "READ")
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Could not open {input_path}")

    summed: Dict[Tuple[float, float], "ROOT.TH2"] = {}
    for key in f_in.GetListOfKeys():
        name = key.GetName()
        m = H2D_NAME_RE.match(name)
        if not m:
            continue
        pt_min = float(m.group("pt_min"))
        pt_max = float(m.group("pt_max"))
        h2 = f_in.Get(name)
        if not h2.InheritsFrom("TH2"):
            continue
        bin_key = (pt_min, pt_max)
        if bin_key in summed:
            summed[bin_key].Add(h2)
        else:
            clone = h2.Clone(f"_sum_{pt_min}_{pt_max}_{id(h2)}")
            clone.SetDirectory(0)
            summed[bin_key] = clone
    f_in.Close()

    if not summed:
        raise RuntimeError(f"No matching h2D_* histograms found in {input_path}")
    return summed


def project_year(
        year_entries: List[Dict],
        output_path: str
) -> Tuple[List[Tuple[float, float]], Tuple[int, int]]:
    """Project per-centrality and centrality-integrated TH2s for one year.

    Writes one ROOT file containing ``h_mass_<PT*10>_<PT*10>_cent_<CMIN>_<CMAX>``
    for every available centrality plus the integrated range
    ``(min(cent_min), max(cent_max))``.

    Returns ``(pt_bins, (integrated_cent_min, integrated_cent_max))``.
    """
    per_cent = {
        (entry["cent_min"], entry["cent_max"]): _load_h2_per_pt(entry["input_path"])
        for entry in year_entries
    }

    pt_bin_sets = {tuple(sorted(h2.keys())) for h2 in per_cent.values()}
    if len(pt_bin_sets) != 1:
        raise RuntimeError(
            "Inconsistent pT bins across centralities for inputs: "
            f"{[e['input_path'] for e in year_entries]}"
        )
    pt_bins = list(next(iter(pt_bin_sets)))

    # Integrated TH2 per pT = sum across all available centralities.
    integrated: Dict[Tuple[float, float], "ROOT.TH2"] = {}
    for h2_per_pt in per_cent.values():
        for pt_key, h2 in h2_per_pt.items():
            if pt_key in integrated:
                integrated[pt_key].Add(h2)
            else:
                clone = h2.Clone(f"_int_{pt_key[0]}_{pt_key[1]}")
                clone.SetDirectory(0)
                integrated[pt_key] = clone

    int_cmin = min(c for c, _ in per_cent)
    int_cmax = max(c for _, c in per_cent)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f_out = ROOT.TFile.Open(output_path, "RECREATE")
    for (cmin, cmax), h2_per_pt in per_cent.items():
        for pt_min, pt_max in pt_bins:
            h1 = h2_per_pt[(pt_min, pt_max)].ProjectionX(
                f"h_mass_{pt_min*10:.0f}_{pt_max*10:.0f}_cent_{cmin}_{cmax}"
            )
            h1.SetDirectory(0)
            f_out.cd()
            h1.Write()
    for pt_min, pt_max in pt_bins:
        h1 = integrated[(pt_min, pt_max)].ProjectionX(
            f"h_mass_{pt_min*10:.0f}_{pt_max*10:.0f}_cent_{int_cmin}_{int_cmax}"
        )
        h1.SetDirectory(0)
        f_out.cd()
        h1.Write()
    f_out.Close()

    return pt_bins, (int_cmin, int_cmax)


def merge_pdfs(cfg):
    """Merge individual PDF files into a single PDF inside this run's fits dir."""
    if "pdf" not in cfg["output"]["formats"]:
        return

    output_dir = os.path.join(os.path.expanduser(cfg["output"]["directory"]), "fits")
    if not os.path.isdir(output_dir):
        return

    fits_out_path = os.path.join(output_dir, "fit_mass_merged.pdf")
    if os.path.exists(fits_out_path):
        os.remove(fits_out_path)

    residuals_out_path = os.path.join(output_dir, "fit_massres_merged.pdf")
    if os.path.exists(residuals_out_path):
        os.remove(residuals_out_path)

    files = os.listdir(output_dir)
    pdf_files = [f for f in files if f.endswith('.pdf') and f.startswith('fit_mass_pt_')]
    pdf_files = sorted(pdf_files, key=lambda x: (
        int(x.split('_')[3]),
        int(x.split('_')[4]) if 'cent' in x else -1
    ))

    pdf_files_residuals = [
        f for f in files if f.endswith('.pdf') and f.startswith('fit_massres_pt_')
    ]
    pdf_files_residuals = sorted(pdf_files_residuals, key=lambda x: (
        int(x.split('_')[3]),
        int(x.split('_')[4]) if 'cent' in x else -1
    ))

    if pdf_files:
        merger = PdfWriter()
        for pdf in pdf_files:
            merger.append(os.path.join(output_dir, pdf))
        merger.write(fits_out_path)
        merger.close()

    if pdf_files_residuals:
        merger = PdfWriter()
        for pdf in pdf_files_residuals:
            merger.append(os.path.join(output_dir, pdf))
        merger.write(residuals_out_path)
        merger.close()


def merge_partial_root(cfg):
    """Merge individual ``*_partial.root`` files into one ROOT file."""
    if "root" not in cfg["output"]["formats"]:
        return

    output_dir = os.path.join(os.path.expanduser(cfg["output"]["directory"]), "fits")
    if not os.path.isdir(output_dir):
        return

    files = os.listdir(output_dir)
    root_files = [f for f in files if f.endswith('_partial.root')]
    if not root_files:
        return

    merged_file_path = os.path.join(output_dir, f"fits_{cfg['output']['suffix']}.root")
    if os.path.exists(merged_file_path):
        os.remove(merged_file_path)

    with uproot.recreate(merged_file_path) as merged_file:
        for root_file in root_files:
            with uproot.open(os.path.join(output_dir, root_file)) as f:
                for key in f.keys():
                    merged_file[key] = f[key]


def run_fit(fit_config: FitConfig) -> Tuple[FitConfig, Dict]:
    """Run a single fit and return ``(config, results)``."""
    fit_handler = FitHandler(fit_config)
    results = fit_handler.get_results()
    return fit_config, results


def get_corr_bkg_config(cfg: Dict, i_pt: int) -> CorrelatedBackgroundConfig:
    """Create a CorrelatedBackgroundConfig object based on the provided configuration."""
    bkg_cfg = cfg["fit_configs"]["bkg"]
    if not bkg_cfg["use_bkg_templ"][i_pt]:
        return None

    bkg_norm = bkg_cfg["templ_norm"]

    return CorrelatedBackgroundConfig(
        fix_to_file=bkg_norm["fix_to_file"][i_pt],
        fix_to_mb=bkg_norm["fix_to_mb"][i_pt],
        fix_with_br=bkg_norm["fix_with_br"][i_pt],
        file_name_for_fix=bkg_norm["file_name_for_fix"],
        hist_name_for_fix=bkg_norm["hist_name_for_fix"],
        backgrounds=[
            CorrelatedBackground(
                name=bkg["name"],
                file_norm=bkg["file_norm"],
                norm_hist_name=bkg["norm_hist_name"],
                template_file=bkg["template_file"],
                template_hist_name=bkg["template_hist_name"],
                br=BRInfo(pdg=bkg["br"]["pdg"], simulations=bkg["br"]["simulations"])
            )
            for bkg in bkg_norm["backgrounds"]
        ],
        signal_norm_file=bkg_norm["signal"]["file_norm"],
        signal_hist_name=bkg_norm["signal"]["hist_name"],
        signal_br=BRInfo(
            pdg=bkg_norm["signal"]["br"]["pdg"],
            simulations=bkg_norm["signal"]["br"]["simulations"]
        )
    )


def get_parameter_setup(
        cfg: List[Dict],
        i_pt: int,
        mb_result: Dict = None,
        result_sigma_fix: Dict = None
    ) -> List[Dict]:
    """Build the per-function parameter setup for a given pT bin."""
    sgn_cfg = cfg["fit_configs"]["signal"]
    param_cfg = sgn_cfg["par_init_limit"]
    functions_setup = []
    for i_func, func_setup in enumerate(param_cfg):
        param_setup = {}
        for par_name, par_values in func_setup.items():
            if par_values["fix_to_mb"][i_pt] and mb_result is not None:
                param_setup[par_name] = {
                    "init": mb_result[par_name][i_func][0],
                    "min": par_values["min"][i_pt],
                    "max": par_values["max"][i_pt],
                    "fix_to_config_value": True,
                    "fix_to_file": False,
                }
            elif par_name == "sigma" \
                and i_func == 1 \
                    and sgn_cfg["fix_sigma_dplus_to_ds"][i_pt] \
                        and result_sigma_fix is not None:
                ratio_sigma_dplus_to_ds = -1.
                if isinstance(sgn_cfg["ratio_sigma_dplus_to_ds"], list):
                    ratio_sigma_dplus_to_ds = sgn_cfg["ratio_sigma_dplus_to_ds"][i_pt]
                elif isinstance(sgn_cfg["ratio_sigma_dplus_to_ds"], (int, float)):
                    ratio_sigma_dplus_to_ds = sgn_cfg["ratio_sigma_dplus_to_ds"]

                param_setup[par_name] = {
                    "init": result_sigma_fix[par_name][0][0] * ratio_sigma_dplus_to_ds,
                    "min": result_sigma_fix[par_name][0][0] * ratio_sigma_dplus_to_ds - 0.01,
                    "max": result_sigma_fix[par_name][0][0] * ratio_sigma_dplus_to_ds + 0.01,
                    "fix_to_config_value": True,
                    "fix_to_file": False
                }
            else:
                param_setup[par_name] = {
                    "init": par_values["init"][i_pt],
                    "min": par_values["min"][i_pt],
                    "max": par_values["max"][i_pt],
                    "fix_to_config_value": par_values["fix_to_config_value"][i_pt],
                    "fix_to_file": par_values["fix_to_file"][i_pt]
                }
        functions_setup.append(param_setup)
    return functions_setup


def get_config(
        cfg: Dict,
        pt_info: Tuple[int, float, float],
        cent_info: Tuple[int, int],
        mb_results: Dict = None,
        result_sigma_fix: Dict = None
) -> FitConfig:
    """Create a FitConfig object for one (pT, centrality) bin."""
    i_pt, pt_min, pt_max = pt_info
    cent_min, cent_max = cent_info

    ratio_sigma_dplus_to_ds = -1.
    if isinstance(cfg["fit_configs"]["signal"]["ratio_sigma_dplus_to_ds"], list):
        ratio_sigma_dplus_to_ds = cfg["fit_configs"]["signal"]["ratio_sigma_dplus_to_ds"][i_pt]
    elif isinstance(cfg["fit_configs"]["signal"]["ratio_sigma_dplus_to_ds"], (int, float)):
        ratio_sigma_dplus_to_ds = cfg["fit_configs"]["signal"]["ratio_sigma_dplus_to_ds"]

    return FitConfig(
        pt_min=pt_min,
        pt_max=pt_max,
        cent_min=cent_min,
        cent_max=cent_max,
        mass_range=[
            cfg["fit_configs"]["mass"]["mins"][i_pt],
            cfg["fit_configs"]["mass"]["maxs"][i_pt]
        ],
        signal_pdfs=cfg["fit_configs"]["signal"]["signal_funcs"][i_pt],
        bkg_pdfs=cfg["fit_configs"]["bkg"]["bkg_funcs"][i_pt],
        rebin=cfg["fit_configs"]["rebin"][i_pt],
        param_setup=get_parameter_setup(cfg, i_pt, mb_results, result_sigma_fix),
        data_path=cfg["inputs"]["data"],
        file_for_params_fix=cfg["fit_configs"]["signal"]["file_for_params_fix"],
        suffix_hist_for_params_fix=cfg["fit_configs"]["signal"]["suffix_hist_for_params_fix"],
        fix_dplus_sigma_to_ds=cfg["fit_configs"]["signal"]["fix_sigma_dplus_to_ds"][i_pt],
        ratio_sigma_dplus_to_ds=ratio_sigma_dplus_to_ds,
        correlated_bkg=get_corr_bkg_config(cfg, i_pt),
        draw_figures=cfg["output"]["save_all_fits"],
        draw_formats=cfg["output"]["formats"],
        output_dir=os.path.join(os.path.expanduser(cfg["output"]["directory"]), "fits"),
        fig_suffix=str(cfg["output"]["suffix"])
    )


def _validate_config_lengths(cfg: Dict, n_pt: int, label: str):
    """Verify that every per-pT array in the config matches the discovered bin count."""
    fc = cfg["fit_configs"]
    sig = fc["signal"]
    bkg = fc["bkg"]
    expected = {
        "fit_configs.mass.mins": fc["mass"]["mins"],
        "fit_configs.mass.maxs": fc["mass"]["maxs"],
        "fit_configs.rebin": fc["rebin"],
        "fit_configs.signal.signal_funcs": sig["signal_funcs"],
        "fit_configs.signal.fix_sigma_dplus_to_ds": sig["fix_sigma_dplus_to_ds"],
        "fit_configs.bkg.bkg_funcs": bkg["bkg_funcs"],
        "fit_configs.bkg.use_bkg_templ": bkg["use_bkg_templ"],
    }
    if isinstance(sig["ratio_sigma_dplus_to_ds"], list):
        expected["fit_configs.signal.ratio_sigma_dplus_to_ds"] = sig["ratio_sigma_dplus_to_ds"]
    for i_func, func_setup in enumerate(sig["par_init_limit"]):
        for par_name, par_values in func_setup.items():
            for sub_key in ("init", "min", "max", "fix_to_config_value",
                            "fix_to_file", "fix_to_mb"):
                expected[
                    f"fit_configs.signal.par_init_limit[{i_func}].{par_name}.{sub_key}"
                ] = par_values[sub_key]

    bad = {k: len(v) for k, v in expected.items() if len(v) != n_pt}
    if bad:
        raise ValueError(
            f"Config has per-pT arrays whose length != discovered {n_pt} bins "
            f"(from {label}):\n  " + "\n  ".join(f"{k}: len={n}" for k, n in bad.items())
        )


def _run_pass(submissions: List[FitConfig], max_workers: int, desc: str) -> Dict:
    """Submit a batch of fits and collect results keyed by (pt_min, pt_max, cent_min, cent_max)."""
    results = {}
    if not submissions:
        return results
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_fit, fc) for fc in submissions]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            fit_cfg, res = future.result()
            results[(fit_cfg.pt_min, fit_cfg.pt_max,
                     fit_cfg.cent_min, fit_cfg.cent_max)] = (fit_cfg, res)
    return results


def fit_one_year(  # pylint: disable=too-many-locals
        base_cfg: Dict,
        year: str,
        year_entries: List[Dict],
        output_dir: str
) -> Tuple[Dict, List[Tuple[float, float]], Tuple[int, int], List[Tuple[int, int]]]:
    """Run the per-year fit pipeline: centrality-integrated fit first, then
    per-centrality fits using those values for any parameter marked ``fix_to_mb``.

    Returns ``(results, pt_bins, integrated_range, cent_ranges)`` where ``results``
    keys are ``(pt_min, pt_max, cent_min, cent_max)``; the integrated fit is
    included alongside the per-centrality entries. ``pt_bins`` is the list of
    ``(pt_min, pt_max)`` tuples discovered in the input files.
    """
    os.makedirs(os.path.join(output_dir, "fits"), exist_ok=True)
    projections_path = os.path.join(output_dir, "projections.root")
    pt_bins, integrated_range = project_year(year_entries, projections_path)
    _validate_config_lengths(base_cfg, len(pt_bins), projections_path)
    pt_mins = [pt_min for pt_min, _ in pt_bins]
    pt_maxs = [pt_max for _, pt_max in pt_bins]

    file_cfg = copy.deepcopy(base_cfg)
    file_cfg["inputs"]["data"] = projections_path
    file_cfg["output"]["directory"] = output_dir

    if "root" in file_cfg["output"]["formats"]:
        uproot.recreate(
            os.path.join(output_dir, "fits", f"fits_{file_cfg['output']['suffix']}.root")
        )

    int_cmin, int_cmax = integrated_range
    sig_cfg = base_cfg["fit_configs"]["signal"]
    max_workers = base_cfg["max_workers"]
    cent_ranges = [(e["cent_min"], e["cent_max"]) for e in year_entries]

    # 1) Centrality-integrated (MB-like) pass.
    integrated_results = _run_pass(
        [get_config(file_cfg, (i, pt_min, pt_max), (int_cmin, int_cmax))
         for i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs))],
        max_workers, f"{year} integrated"
    )

    # 2) Integrated re-fit with D+ sigma fixed to ratio * Ds sigma where requested.
    integrated_results.update(_run_pass(
        [get_config(file_cfg, (i, pt_min, pt_max), (int_cmin, int_cmax),
                    None, integrated_results[(pt_min, pt_max, int_cmin, int_cmax)][1])
         for i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs))
         if sig_cfg["fix_sigma_dplus_to_ds"][i]],
        max_workers, f"{year} integrated (dplus sigma)"
    ))

    # 3) Per-centrality fits using integrated results to fix params (fix_to_mb).
    per_cent_subs = []
    for cent_min, cent_max in cent_ranges:
        for i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
            mb_res = integrated_results[(pt_min, pt_max, int_cmin, int_cmax)][1]
            per_cent_subs.append(
                get_config(file_cfg, (i, pt_min, pt_max), (cent_min, cent_max), mb_res)
            )
    cent_results = _run_pass(per_cent_subs, max_workers, f"{year} per-cent")

    # 4) Per-centrality re-fit with D+ sigma fixed where it wasn't already fix_to_mb.
    refit_subs = []
    for (pt_min, pt_max, cent_min, cent_max), (_, res) in cent_results.items():
        i_pt = pt_mins.index(pt_min)
        if not sig_cfg["fix_sigma_dplus_to_ds"][i_pt]:
            continue
        if not any(
            "sigma" in fp and not fp["sigma"]["fix_to_mb"][i_pt]
            for fp in sig_cfg["par_init_limit"]
        ):
            continue
        refit_subs.append(get_config(
            file_cfg, (i_pt, pt_min, pt_max), (cent_min, cent_max), None, res
        ))
    cent_results.update(_run_pass(refit_subs, max_workers, f"{year} per-cent (dplus sigma)"))

    merge_pdfs(file_cfg)
    merge_partial_root(file_cfg)

    cent_results.update(integrated_results)
    return cent_results, pt_bins, integrated_range, cent_ranges


def fit(config_file_name):
    """Run the per-year fit pipeline for every year discovered under ``inputs.data_dir``."""
    with open(config_file_name, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_output_dir = os.path.expanduser(cfg["output"]["directory"])
    os.makedirs(base_output_dir, exist_ok=True)

    by_year = discover_input_files(cfg["inputs"]["data_dir"])
    if not by_year:
        raise RuntimeError(
            f"No MassDistributions.root files found under {cfg['inputs']['data_dir']}"
        )

    all_outputs = []
    for year in sorted(by_year):
        year_entries = sorted(
            by_year[year], key=lambda e: (e["cent_min"], e["cent_max"])
        )
        output_dir = os.path.join(base_output_dir, year)
        os.makedirs(output_dir, exist_ok=True)

        results, pt_bins, integrated_range, cent_ranges = fit_one_year(
            cfg, year, year_entries, output_dir
        )
        pt_mins = [pt_min for pt_min, _ in pt_bins]
        pt_maxs = [pt_max for _, pt_max in pt_bins]

        rows = []
        for fit_config, result in results.values():
            row = result.copy()
            row.update(fitconfig_to_dict(fit_config))
            row["year"] = year
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(
            os.path.join(output_dir, f"fit_results{cfg['output']['suffix']}.parquet"),
            index=False
        )

        # HistHandler needs every (cent_min, cent_max) seen in the dataframe.
        cent_mins = [c[0] for c in cent_ranges] + [integrated_range[0]]
        cent_maxs = [c[1] for c in cent_ranges] + [integrated_range[1]]
        h_handler = HistHandler(pt_mins, pt_maxs, cent_mins, cent_maxs)
        h_handler.set_histos(df)
        h_handler.dump_to_root(
            os.path.join(output_dir, f"mass_fits{cfg['output']['suffix']}.root")
        )

        all_outputs.append(df)

    if all_outputs:
        pd.concat(all_outputs, ignore_index=True).to_parquet(
            os.path.join(base_output_dir, f"fit_results{cfg['output']['suffix']}.parquet"),
            index=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('config_file_name', metavar='text', default='')
    args = parser.parse_args()

    fit(args.config_file_name)

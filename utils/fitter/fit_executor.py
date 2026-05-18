"""
Low-level fit execution:
fitter creation, parameter handling, correlated background, fit execution.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from flarefly.fitter import F2MassFitter
from flarefly.data_handler import DataHandler

PDG_IDS = {"ds": 431, "dplus": 411}

SIGNAL_DEFAULTS = {
    "gaussian": {
        "mu": {"init": None, "min": None, "max": None, "fix": False},
        "sigma": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False}
    },
    "doublegaus": {
        "mu": {"init": None, "min": None, "max": None, "fix": False},
        "sigma1": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False},
        "sigma2": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False},
        "frac1": {"init": 0.01, "min": 0.0, "max": 1.0, "fix": False}
    },
    "doublecb": {
        "mu": {"init": None, "min": None, "max": None, "fix": False},
        "sigma": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False},
        "alphar": {"init": 0.5, "min": 0.0, "max": 10.0, "fix": False},
        "alphal": {"init": 0.5, "min": 0.0, "max": 10.0, "fix": False},
        "nl": {"init": 1.0, "min": 0.0, "max": 10.0, "fix": False},
        "nr": {"init": 1.0, "min": 0.0, "max": 10.0, "fix": False}
    },
    "doublecbsymm": {
        "mu": {"init": None, "min": None, "max": None, "fix": False},
        "sigma": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False},
        "alpha": {"init": 5.0, "min": 0.5, "max": 10.0, "fix": False},
        "n": {"init": 10.0, "min": 5.0, "max": 100.0, "fix": False}
    },
    "genergausexptailsymm": {
        "mu": {"init": None, "min": None, "max": None, "fix": False},
        "sigma": {"init": 0.01, "min": 0.001, "max": 0.03, "fix": False},
        "alpha": {"init": 3.0, "min": 0.0, "max": 10.0, "fix": False}
    }
}

BACKGROUND_DEFAULTS = {
    "expo": {
        "lam": {"init": -2.0, "min": -10.0, "max": 10.0, "fix": False}
    },
    "chebpol2": {
        "c0": {"init": 0.6, "min": -1.0, "max": 1.0, "fix": False},
        "c1": {"init": -0.2, "min": -2.0, "max": 2.0, "fix": False},
        "c2": {"init": 0.01, "min": -1.0, "max": 1.0, "fix": False}
    },
    "chebpol3": {
        "c0": {"init": 0.4, "min": 0.0, "max": 1.0, "fix": False},
        "c1": {"init": -0.2, "min": -2.0, "max": 2.0, "fix": False},
        "c2": {"init": -0.01, "min": -2.0, "max": 2.0, "fix": False},
        "c3": {"init": 0.01, "min": -1.0, "max": 1.0, "fix": False}
    }
}

class FitExecutor:  # pylint: disable=too-many-instance-attributes
    """
    Low-level fit execution: create fitter, handle parameters, correlated bkg, execute fit.
    """

    def __init__(
            self,
            data: DataHandler,
            signal_functions: List[str],
            background_functions: List[str],
            name: str
        ):
        self._data = data
        self._signal_functions = signal_functions
        self._bkg_functions = background_functions
        self._name = name

        self._fitter = None

        self._has_correlated_bkg = False
        self._correlated_bkg_data_hdl = []
        self._fix_correlated_bkg = []
        self._correlated_bkg_norm = []
        self._correlated_bkg_labels = []
        self._fix_correlated_bkg_to_sng = []

        self._sgn_params = [{} for _ in signal_functions]
        self._bkg_params = [{} for _ in background_functions]

    def set_correlated_background(
        self,
        correlated_bkg: DataHandler,
        fix: bool = False,
        norm: float = 0.0,
        name: str = "",
        fix_to_signal: int = 1
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Set correlated background data handler and options."""
        self._correlated_bkg_data_hdl.append(correlated_bkg)
        self._has_correlated_bkg = True
        self._fix_correlated_bkg.append(fix)
        self._correlated_bkg_norm.append(norm)
        if name != "":
            self._correlated_bkg_labels.append(name)
        else:
            self._correlated_bkg_labels.append(
                f"correlated background {len(self._correlated_bkg_data_hdl)}"
            )
        self._fix_correlated_bkg_to_sng.append(fix_to_signal)

    def set_parameter(
        self,
        is_signal: bool,
        index: int,
        name: str,
        value: float,
        minv: float,
        maxv: float,
        fix: bool
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Set parameter for signal or background function."""
        param_dict = self._sgn_params if is_signal else self._bkg_params
        param_dict[index][name] = {
            "value": value,
            "min": minv,
            "max": maxv,
            "fix": fix
        }

    def execute(self) -> Tuple[Dict, Optional[F2MassFitter]]:
        """
        Execute a single fit.
        
        Returns:
            (results_dict, fitter_instance)
        """

        # Create fitter
        self._create_fitter()

        # Set parameters
        self._initialize_parameters()

        # Handle correlated backgrounds
        self._setup_correlated_backgrounds()

        # Do fit
        fit_result = self._fitter.mass_zfit()

        # Extract results
        results = self._extract_results(fit_result.converged)

        return results, self._fitter



    def _create_fitter(self) -> F2MassFitter:
        """Create fitter."""

        name_background_pdf = ["hist" for _ in self._correlated_bkg_data_hdl] + self._bkg_functions

        label_signal_pdfs = [r"$\mathrm{D_{s}^{+}}$ signal", r"$\mathrm{D^{+}}$ signal"]
        label_bkg_pdfs = self._correlated_bkg_labels + ["Combinatorial background"]

        self._fitter = F2MassFitter(
            self._data,
            name_signal_pdf=self._signal_functions,
            name_background_pdf=name_background_pdf,
            name=self._name,
            chi2_loss=True,
            label_signal_pdf=label_signal_pdfs,
            label_bkg_pdf=label_bkg_pdfs,
            verbosity=7,
            tol=1.e-1
        )

    def _get_param_settings(self, idx, param, defaults, is_signal: bool):
        """Get parameter settings from config or defaults."""
        if is_signal:
            param_info = self._sgn_params[idx].get(param, {})
        else:
            param_info = self._bkg_params[idx].get(param, {})
        value = param_info.get("value", defaults["init"])
        min_val = param_info.get("min", defaults["min"])
        max_val = param_info.get("max", defaults["max"])
        fix = param_info.get("fix", defaults["fix"])
        return param, value, min_val, max_val, fix

    def _initialize_parameters(
        self
    ):
        """Initialize all signal and background parameters."""

        # Signal parameters
        for idx, signal_func in enumerate(self._signal_functions):
            particle = "ds" if idx == 0 else "dplus"
            self._fitter.set_particle_mass(idx, pdg_id=PDG_IDS[particle])
            if signal_func in SIGNAL_DEFAULTS:
                for name, defaults in SIGNAL_DEFAULTS[signal_func].items():
                    param, value, min_val, max_val, fix = self._get_param_settings(
                        idx, name, defaults, True
                    )
                    if value:
                        self._fitter.set_signal_initpar(
                            idx, param, value, limits=[min_val, max_val], fix=fix
                        )

        # Background parameters
        for idx, bkg_func in enumerate(self._bkg_functions):
            if bkg_func in BACKGROUND_DEFAULTS:
                for name, defaults in BACKGROUND_DEFAULTS[bkg_func].items():
                    param, value, min_val, max_val, fix = self._get_param_settings(
                        idx, name, defaults, False
                    )
                    self._fitter.set_background_initpar(
                        len(self._correlated_bkg_data_hdl) + idx,
                        param,
                        value,
                        limits=[min_val, max_val], fix=fix
                    )

    def _setup_correlated_backgrounds(self):
        """Setup correlated backgrounds in the fitter."""
        if self._has_correlated_bkg:
            for i, bkg_hdl in enumerate(self._correlated_bkg_data_hdl):
                self._fitter.set_background_template(i, bkg_hdl)
                if self._fix_correlated_bkg[i]:
                    self._fitter.fix_bkg_frac_to_signal_pdf(
                        i,
                        self._fix_correlated_bkg_to_sng[i],
                        self._correlated_bkg_norm[i]
                    )

    def _extract_results(self, converged: bool) -> Dict:
        """Extract fit results into dictionary."""
        n_signal = len(self._signal_functions)
        if converged:
            results = {
                "raw_yields": [list(self._fitter.get_raw_yield(i)) for i in range(n_signal)],
                "mu": [list(self._fitter.get_mass(i)) for i in range(n_signal)],
                "chi2": float(self._fitter.get_chi2_ndf()),
                "significance": [list(self._fitter.get_significance(i)) for i in range(n_signal)],
                "signal": [list(self._fitter.get_signal(i)) for i in range(n_signal)],
                "background": [list(self._fitter.get_background(i)) for i in range(n_signal)],
                "converged": converged
            }

            # Add signal-specific parameters
            self._add_signal_params_to_results(results, n_signal)

            # Add background fractions
            try:
                fracs = self._fitter._F2MassFitter__get_all_fracs()  # pylint: disable=protected-access
                results["fracs"] = list(fracs)
                results.update(self._extract_bkg_fractions(fracs))
            except Exception:
                results["fracs"] = None
        else:
            results = self._create_empty_result(n_signal)

        return results
    
    def extend_results_bincounting(self, results: Dict, nsigmas: List[float]):
        """
        Extend the results dictionary with bin counting information for given nsigma values.
        """
        n_signal = len(self._signal_functions)
        for nsigma in nsigmas:
            results[f"raw_yields_bincounting_{nsigma}"] = [self._fitter.get_raw_yield_bincounting(i, nsigma=nsigma) for i in range(n_signal)]
        return results


    def _add_signal_params_to_results(
        self,
        results: Dict,
        n_signal: int
    ):
        """Add signal function specific parameters to results."""
        param_lists = {
            func_type: list(parameters.keys())
            for func_type, parameters in SIGNAL_DEFAULTS.items()
        }

        for i, func in enumerate(self._signal_functions):
            for param in param_lists.get(func, []):
                if param not in results:
                    results[param] = [None] * n_signal
                results[param][i] = self._fitter.get_signal_parameter(i, param)

    def _extract_bkg_fractions(self, fracs: Tuple) -> Dict:
        """Extract background fraction information."""
        sig_fracs, bkg_fracs, _, _, sig_errs, bkg_errs = fracs
        results = {}

        if len(bkg_fracs) > 0:
            for i, (frac, err) in enumerate(zip(bkg_fracs[:-1], bkg_errs[:-1])):
                results[f"corr_bkg_frac_{i}"] = [frac, err]
                if len(sig_fracs) > 1:
                    ratio = frac / sig_fracs[1]
                    ratio_err = ratio * np.sqrt((err/frac)**2 + (sig_errs[1]/sig_fracs[1])**2)
                    results[f"corr_bkg_frac_over_dplus_{i}"] = [ratio, ratio_err]

        return results

    def _create_empty_result(self, n_signal: int) -> Dict:
        """Create empty result for failed fits."""
        return {
            "raw_yields": [None] * n_signal,
            "sigma": [None] * n_signal,
            "mean": [None] * n_signal,
            "chi2": None,
            "significance": [None] * n_signal,
            "signal": [None] * n_signal,
            "background": [None] * n_signal,
            "fracs": None,
            "converged": False
        }

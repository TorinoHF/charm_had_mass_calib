"""Module for handling output of fitting."""

import dataclasses
import numpy as np
import ROOT
import uproot

@dataclasses.dataclass
class BinsHelper:
    """
    Helper class for binning.

    Parameters:
    - mins (list or array-like): A list or array of minimum values for each bin.
    - maxs (list or array-like): A list or array of maximum values for each bin.

    Attributes:
    - mins (list or array-like): Stores the minimum values for each bin.
    - maxs (list or array-like): Stores the maximum values for each bin.
    - bins (list of tuples): A list of tuples where each tuple contains
        the minimum and maximum values for a bin.
    - edges (numpy.ndarray): An array of bin edges.
    - n_bins (int): The number of bins.
    """
    mins: list
    maxs: list

    def __post_init__(self):
        self.bins = [*zip(self.mins, self.maxs)]
        self.edges = np.asarray(self.mins + [self.maxs[-1]], 'd')
        self.n_bins = len(self.mins)


class HistHandler:  # pylint: disable=too-many-instance-attributes
    """
    Class designed to handle the creation, manipulation, and storage
    of histograms for various observables.

    Parameters:
    - pt_mins (list or array-like): Minimum values for pT bins.
    - pt_maxs (list or array-like): Maximum values for pT bins.
    - cent_mins (list or array-like, optional): Minimum values for centrality bins.
    - cent_maxs (list or array-like, optional): Maximum values for centrality bins.
    """

    def __init__(self, pt_mins, pt_maxs, cent_mins=None, cent_maxs=None):
        self._pt_info = BinsHelper(pt_mins, pt_maxs)
        self._cent_info = BinsHelper(cent_mins, cent_maxs) if cent_mins is not None else None
        self._pt_axis_title = '#it{p}_{T} (GeV/#it{c})'
        self._n_ev = None
        self._histos = {}
        self._h_collisions = None

        self._observable_config = self._get_observable_config()
        self._build_histos()

    def _get_observable_config(self):
        """Get configuration for all observables and their axis titles."""
        return {
            # Common observables (have both _sig1 and _sig2 versions)
            "common": {
                "raw_yields": "Raw yields",
                "sigma": "Width (GeV/#it{c}^{2})",
                "sigma1": "Width_{1} (GeV/#it{c}^{2})",
                "sigma2": "Width_{2} (GeV/#it{c}^{2})",
                "frac1": "Gaussian fraction",
                "mu": "Mean (GeV/#it{c}^{2})",
                "raw_yield_over_ev": "Raw yields / N_{ev}",
                "significance": "Significance (3#sigma)",
                "significance_over_sqrt_ev": "Significance / #sqrt{N_{ev}}",
                "s_over_b": "S/B (3#sigma)",
                "signal": "Signal (3#sigma)",
                "background": "Background (3#sigma)",
                "alphal": "#alpha_{l}",
                "alphar": "#alpha_{r}",
                "nl": "n_{l}",
                "nr": "n_{r}",
                "alpha": "#alpha",
                "n": "n"
            },
            # Non-common observables (single version only)
            "not_common": {
                "chi2": "#chi^{2}/#it{ndf}",
                "sigma_ratio_second_first_peak": "Width second peak / width first peak",
                "corr_bkg_frac_over_dplus_0": "Corr. bkg / D^{+} signal",
                "corr_bkg_frac_0": "Corr. bkg fraction"
            }
        }

    def _create_histo_name(self, obs, cent_min=None, cent_max=None):
        """Generate histogram name based on observable and centrality."""
        base_name = f'h_{obs}'
        if cent_min is not None and cent_max is not None:
            base_name += f'_{cent_min:.0f}_{cent_max:.0f}'
        return base_name

    def _create_histogram(self, obs, y_title):
        """Create histograms for a given observable."""
        histos = []

        if self._cent_info is None:
            hist = ROOT.TH1D(
                self._create_histo_name(obs),
                f';{self._pt_axis_title};{y_title}',
                self._pt_info.n_bins, self._pt_info.edges
            )
            hist.SetDirectory(0)
            histos.append(hist)
        else:
            for cent_min, cent_max in zip(self._cent_info.mins, self._cent_info.maxs):
                hist = ROOT.TH1D(
                    self._create_histo_name(obs, cent_min, cent_max),
                    f';{self._pt_axis_title};{y_title}',
                    self._pt_info.n_bins, self._pt_info.edges
                )
                hist.SetDirectory(0)
                histos.append(hist)

        return histos

    def _build_histos(self):
        """Build histograms for all observables."""
        # Create common observables (both _sig1 and _sig2 versions)
        for obs, y_title in self._observable_config["common"].items():
            self._histos[f"{obs}_sig1"] = self._create_histogram(f"{obs}_sig1", y_title)
            self._histos[f"{obs}_sig2"] = self._create_histogram(f"{obs}_sig2", y_title)

        # Create non-common observables
        for obs, y_title in self._observable_config["not_common"].items():
            self._histos[obs] = self._create_histogram(obs, y_title)

    def set_n_ev(self, n_ev):
        """Set the number of events."""
        self._n_ev = n_ev

    def set_h_collisions(self, h_collisions):
        """Set the histogram for collisions."""
        self._h_collisions = h_collisions

    def _get_centrality_index(self, row):
        """Get centrality index from row data."""
        if "cent_min_cfg" in row and "cent_max_cfg" in row:
            cent_tuple = (row["cent_min_cfg"], row["cent_max_cfg"])
            return self._cent_info.bins.index(cent_tuple)
        return 0

    def _set_histogram_values(self, hist, i_pt, value, error=None):
        """Set histogram bin content and error."""
        hist.SetBinContent(i_pt + 1, value)
        if error is not None:
            hist.SetBinError(i_pt + 1, error)

    def _get_available_observables(self, row):
        """Determine which observables are available in the row data."""
        observables = ["raw_yields", "mu", "significance", "signal", "background"]

        # Check sigma variants
        if "sigma" in row:
            observables.append("sigma")
        else:
            observables.extend(["sigma1", "sigma2", "frac1"])

        # Check Crystal Ball parameters
        if "alphal" in row:
            observables.extend(["alphal", "alphar", "nl", "nr"])
        if "alpha" in row:
            observables.extend(["alpha", "n"])

        return observables

    def _process_common_observables(self, row, i_pt, i_cent, observables):
        """Process common observables (those with _sig1 and _sig2 versions)."""
        for obs in observables:
            if obs not in row:
                continue

            # Set sig1 values
            self._set_histogram_values(
                self._histos[f"{obs}_sig1"][i_cent], i_pt,
                row[obs][0][0], row[obs][0][1]
            )

            # Set sig2 values if available
            if len(row[obs]) > 1:
                self._set_histogram_values(
                    self._histos[f"{obs}_sig2"][i_cent], i_pt,
                    row[obs][1][0], row[obs][1][1]
                )

        # Signal over background
        for idx, suffix in enumerate(["_sig1", "_sig2"]):
            if (len(row["signal"]) > idx and len(row["background"]) > idx and
                row["background"][idx][0] != 0):
                value = row["signal"][idx][0] / row["background"][idx][0]
                error = row["signal"][idx][1] / row["background"][idx][0]
                self._set_histogram_values(
                    self._histos[f"s_over_b{suffix}"][i_cent],
                    i_pt, value, error
                )

        # Sigma ratio (second peak / first peak)
        if ("sigma" in row and len(row["sigma"]) > 1 and
            row["sigma"][0][0] != 0):
            value = row["sigma"][1][0] / row["sigma"][0][0]
            self._set_histogram_values(
                self._histos["sigma_ratio_second_first_peak"][i_cent],
                i_pt, value
            )

    def _process_not_common_observables(self, row, i_pt, i_cent):
        """Process non-common observables."""
        # Chi2
        if "chi2" in row:
            value = row["chi2"] if not isinstance(row["chi2"], (tuple, list)) else row["chi2"][0]
            error = None if not isinstance(row["chi2"], (tuple, list)) else row["chi2"][1]
            self._set_histogram_values(self._histos["chi2"][i_cent], i_pt, value, error)

        # Correlated background observables
        corr_bkg_cols = [col for col in row.keys() if "corr_bkg_frac" in col and "_cfg" not in col]
        for obs in corr_bkg_cols:
            if obs in self._histos:
                value = row[obs] if not isinstance(row[obs], (tuple, list)) else row[obs][0]
                error = None if not isinstance(row[obs], (tuple, list)) else row[obs][1]
                self._set_histogram_values(self._histos[obs][i_cent], i_pt, value, error)

    def set_histos(self, df):
        """Set histogram bin contents and errors based on the provided DataFrame."""
        for _, row in df.iterrows():
            i_pt = self._pt_info.mins.index(row["pt_min_cfg"])
            i_cent = self._get_centrality_index(row)

            # Get available observables for this row
            available_observables = self._get_available_observables(row)

            # Process different types of observables
            self._process_common_observables(row, i_pt, i_cent, available_observables)
            self._process_not_common_observables(row, i_pt, i_cent)

    def dump_to_root(self, output_file):
        """Dump histograms to a ROOT file, one TDirectory per observable."""
        with ROOT.TFile(output_file, "RECREATE") as outfile:
            for key, histos in self._histos.items():
                directory = outfile.mkdir(key)
                directory.cd()
                for hist in histos:
                    hist.Write()
            outfile.cd()

    @property
    def obs_common(self):
        """Get list of common observable names for backward compatibility."""
        return list(self._observable_config["common"].keys())

    @property
    def axes_titles_common(self):
        """Get list of common observable axis titles for backward compatibility."""
        return list(self._observable_config["common"].values())

    @property
    def obs_not_common(self):
        """Get list of non-common observable names for backward compatibility."""
        return list(self._observable_config["not_common"].keys())

    @property
    def axes_titles_not_common(self):
        """Get list of non-common observable axis titles for backward compatibility."""
        return list(self._observable_config["not_common"].values())

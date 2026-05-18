"""
FitHandler class to manage fitting configurations and parameter setups
for Ds and D+ mass fits with correlated backgrounds.
"""
import dataclasses
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import uproot

from flarefly import DataHandler
from fit_executor import FitExecutor

@dataclasses.dataclass(frozen=True)
class CorrelatedBackgroundData:
    """Data class to hold correlated background information."""
    name: str
    template: DataHandler
    norm_to_dplus: Optional[float] = None

@dataclasses.dataclass(frozen=True)
class BRInfo:
    """Data class to hold branching ratio information."""
    pdg: float
    simulations: float

@dataclasses.dataclass(frozen=True)
class CorrelatedBackground:
    """Data class to hold correlated background configuration."""
    name: str
    file_norm: str
    norm_hist_name: str
    template_file: str
    template_hist_name: str
    br: BRInfo

@dataclasses.dataclass(frozen=True)
class CorrelatedBackgroundConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for correlated backgrounds in the fit."""
    fix_to_file: bool
    fix_to_mb: bool
    fix_with_br: bool

    file_name_for_fix: Optional[str]
    hist_name_for_fix: Optional[str]

    backgrounds: List[CorrelatedBackground]

    signal_norm_file: str
    signal_hist_name: str
    signal_br: BRInfo

    # To be set only by FitHandler, for fixing to MB fit value
    fix_to_value: Optional[float] = None
    value_for_fix: Optional[float] = None

@dataclasses.dataclass(frozen=True)
class FitConfig:  # pylint: disable=too-many-instance-attributes
    """A standardized config object that FitHandler understands."""
    pt_min: float
    pt_max: float
    cent_min: List[int] | None
    cent_max: List[int] | None
    mass_range: List[float]
    signal_pdfs: List[str]
    bkg_pdfs: List[str]
    rebin: int
    # Each dict corresponds to a function, containing dicts of parameter settings
    # The keys of the outer dict are the parameter names, while the inner dicts contain:
    # "init": float, "min": float, "max": float,
    # "fix_to_config_value" : bool, "fix_to_file": bool
    param_setup: List[Dict[str, Dict]]
    data_path: str
    file_for_params_fix: Optional[str] = None
    suffix_hist_for_params_fix: Optional[str] = None
    fix_dplus_sigma_to_ds: bool = False
    ratio_sigma_dplus_to_ds: float = 1.0
    correlated_bkg: Optional[CorrelatedBackgroundConfig] = None
    draw_figures: bool = False
    draw_formats: List[str] = dataclasses.field(default_factory=lambda: ["png", "pdf"])
    output_dir: str = ""
    fig_suffix: str = ""
    nsigma_bincounting: List[float] = dataclasses.field(default_factory=list)

class FitHandler():  # pylint: disable=too-few-public-methods
    """Class to handle fitting configurations and parameter setups for a single pT bin."""

    def __init__(self, cfg: FitConfig):
        self._cfg = cfg
        self._result = None
        self._fitter = None
        self._execute()

    def get_results(self) -> Dict[Tuple[Optional[float], Optional[float]], Dict]:
        """Returns the fit results for all centrality bins."""
        return self._result

    def _get_data_handler(self) -> DataHandler:
        """Creates the DataHandler for the specific bin."""

        # Construct histogram name
        pt_cent_suffix = f"{self._cfg.pt_min*10:.0f}_{self._cfg.pt_max*10:.0f}"
        if self._cfg.cent_min is not None and self._cfg.cent_max is not None:
            pt_cent_suffix += f"_cent_{self._cfg.cent_min:.0f}_{self._cfg.cent_max:.0f}"

        hist_name = f"h_mass_{pt_cent_suffix}"

        file_path = self._cfg.data_path

        return DataHandler(
            data=file_path,
            histoname=hist_name,
            limits=self._cfg.mass_range,
            rebin=self._cfg.rebin
        )

    def _execute(self):
        """Main execution method to prepare fit executors for all centrality bins."""
        self._check_parameters()

        executor = self._prepare_executor()
        self._result, self._fitter = executor.execute()

        if self._cfg.nsigma_bincounting:
            self._result = executor.extend_results_bincounting(self._result, self._cfg.nsigma_bincounting)

        if self._cfg.draw_figures:
            self._draw_figures()

    # pylint: disable=too-many-arguments
    def _add_info_on_canvas(self, axs, loc: str):
        """
        Helper method to add text on flarefly mass fit plot

        Parameters
        ----------
        - axs: matplotlib.figure.Axis
            Axis instance of the mass fit figure

        - loc: str
            Location of the info on the figure
        """
        xspace = " "
        text = xspace
        chi2 = self._fitter.get_chi2()
        ndf = self._fitter.get_ndf()
        text += fr"$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf} $\simeq$ {chi2/ndf:.2f}""\n"

        text += "\n\n"
        text += xspace
        text += fr"{self._cfg.pt_min:.0f} < $p_{{\mathrm{{T}}}}$ < {self._cfg.pt_max:.0f} "
        text += r"GeV/$c$, $|y|$ < 0.5""\n"
        if self._cfg.cent_min is not None and self._cfg.cent_max is not None:
            text += xspace + fr"{self._cfg.cent_min:.0f}% < Cent. < {self._cfg.cent_max:.0f}%""\n"

        anchored_text = AnchoredText(text, loc=loc, frameon=False)
        axs.add_artist(anchored_text)

    def _draw_figures(self):
        loc = ["lower left", "upper left"]
        ax_title = r"$M(\mathrm{KK\pi})$ GeV$/c^2$"
        fig, ax = self._fitter.plot_mass_fit(
            style="ATLAS",
            show_extra_info=True,
            figsize=(8, 8), extra_info_loc=loc,
            legend_loc="upper right",
            axis_title=ax_title
        )
        self._add_info_on_canvas(ax, "center right")
        figres = self._fitter.plot_raw_residuals(
            figsize=(8, 8), style="ATLAS",
            extra_info_loc=loc, axis_title=ax_title
        )

        for frmt in self._cfg.draw_formats:
            suffix = f"_{self._cfg.pt_min * 10:.0f}_{self._cfg.pt_max * 10:.0f}_"
            if self._cfg.cent_min is not None and self._cfg.cent_max is not None:
                suffix += f"cent_{self._cfg.cent_min:.0f}_{self._cfg.cent_max:.0f}_"
            suffix += self._cfg.fig_suffix

            if frmt == "root":
                self._fitter.dump_to_root(
                    f"{self._cfg.output_dir}/fits_{suffix}_partial.root",
                    option="recreate", suffix=suffix, num=5000
                )
            else:
                fig.savefig(f"{self._cfg.output_dir}/fit_mass_pt{suffix}.{frmt}")
                figres.savefig(f"{self._cfg.output_dir}/fit_massres_pt{suffix}.{frmt}")
            plt.close(fig)
            plt.close(figres)


    def _check_parameters(self):
        """Validates the configuration parameters."""
        for idx, func in enumerate(self._cfg.signal_pdfs):
            p_cfg = self._cfg.param_setup[idx]
            for param_name, settings in p_cfg.items():
                if settings["min"] >= settings["max"]:
                    raise ValueError(
                        f"Invalid parameter limits for {param_name} in signal PDF {func} "
                        f"at index {idx}: min {settings['min']} >= max {settings['max']}"
                    )
                if not settings["min"] <= settings["init"] <= settings["max"]:
                    raise ValueError(
                        f"Initial value for {param_name} in signal PDF {func} "
                        f"at index {idx} is out of bounds: init {settings['init']} "
                        f"not in [{settings['min']}, {settings['max']}]"
                    )
                if settings["fix_to_config_value"] and settings["fix_to_file"]:
                    raise ValueError(
                        "Multiple fixing options are not allowed "
                        f"for {param_name} in signal PDF {func} at index {idx}."
                    )

    def _prepare_executor(self) -> FitExecutor:
        """
        Main factory method.
        Builds a fully configured FitExecutor for the requested centrality bin.
        """

        data_hdl = self._get_data_handler()

        fit_name = f"ds_pt_{self._cfg.pt_min*10:.0f}_{self._cfg.pt_max*10:.0f}"
        if self._cfg.cent_min is not None and self._cfg.cent_max is not None:
            fit_name += f"_cent_{self._cfg.cent_min:.0f}_{self._cfg.cent_max:.0f}"

        executor = FitExecutor(data_hdl, self._cfg.signal_pdfs, self._cfg.bkg_pdfs, fit_name)

        self._set_parameters(executor)

        self._setup_correlated_backgrounds(executor)

        return executor


    def _set_parameters(self, executor: FitExecutor):
        """Set up parameters for signal and background based on configuration."""

        for idx in range(len(self._cfg.signal_pdfs)):
            particle = "sig1" if idx == 0 else "sig2"

            p_cfg = self._cfg.param_setup[idx]

            for param_name, settings in p_cfg.items():
                is_fixed = settings["fix_to_config_value"] or settings["fix_to_file"]
                val = settings["init"]
                if settings["fix_to_file"]:
                    # Read from file
                    with uproot.open(self._cfg.file_for_params_fix) as f:
                        hist_suffix = self._cfg.suffix_hist_for_params_fix.format(
                            cent_min=self._cfg.cent_min,
                            cent_max=self._cfg.cent_max
                        )
                        hist_name = f'h_{param_name}_{particle}{hist_suffix}'
                        i_pt = np.where(
                            np.isclose(f[hist_name].axis().edges(), self._cfg.pt_min)
                        )[0][0]
                        val = f[hist_name].values()[i_pt]

                executor.set_parameter(
                    is_signal=True, index=idx, name=param_name,
                    value=val, minv=settings["min"], maxv=settings["max"],
                    fix=is_fixed
                )

    def _setup_correlated_backgrounds(
        self,
        executor: FitExecutor
    ):
        """Setup correlated background templates and normalizations."""
        bkg_cfg = self._cfg.correlated_bkg
        if bkg_cfg is None:
            return

        if sum([bkg_cfg.fix_to_file, bkg_cfg.fix_to_mb, bkg_cfg.fix_with_br]) > 1:
            raise ValueError(
                "Multiple fixing options are not allowed for correlated backgrounds."
            )

        corr_backgrounds = self._load_backgrounds()

        for bkg in corr_backgrounds:
            # Load template
            executor.set_correlated_background(
                bkg.template,
                fix = bkg.norm_to_dplus is not None,
                norm = bkg.norm_to_dplus,
                name = bkg.name
            )

    def _load_backgrounds(self) -> List[CorrelatedBackgroundData]:
        """Load correlated background configurations."""
        bkg_cfg = self._cfg.correlated_bkg
        backgrounds = []
        for bkg in bkg_cfg.backgrounds:
            backgrounds.append(
                CorrelatedBackgroundData(
                    name=bkg.name,
                    template=DataHandler(
                        data=bkg.template_file,
                        histoname=bkg.template_hist_name.format(
                            pt_min=f"{self._cfg.pt_min*10:.0f}",
                            pt_max=f"{self._cfg.pt_max*10:.0f}",
                            cent_min=self._cfg.cent_min,
                            cent_max=self._cfg.cent_max
                        ),
                        limits=self._cfg.mass_range,
                        rebin=self._cfg.rebin
                    ),
                    norm_to_dplus=self._get_corr_bkg_norm(bkg)
                )
            )

        return backgrounds

    def _get_corr_bkg_norm(self, bkg: CorrelatedBackground) -> Optional[float]:
        """Get correlated background normalization from file if specified."""
        if self._cfg.correlated_bkg.fix_to_value:
            return self._cfg.correlated_bkg.value_for_fix

        if self._cfg.correlated_bkg.fix_to_file:
            # Read from file
            with uproot.open(self._cfg.correlated_bkg.file_name_for_fix) as f:
                hist_name = self._cfg.correlated_bkg.hist_name_for_fix.format(
                    pt_min=f"{self._cfg.pt_min*10:.0f}",
                    pt_max=f"{self._cfg.pt_max*10:.0f}",
                    cent_min=self._cfg.cent_min,
                    cent_max=self._cfg.cent_max
                )
                i_pt = np.where(np.isclose(f[hist_name].axis().edges(), self._cfg.pt_min))[0][0]
                return f[hist_name].values()[i_pt]

        if self._cfg.correlated_bkg.fix_with_br:
            data_signal_reference = DataHandler(
                data=self._cfg.correlated_bkg.signal_norm_file,
                histoname=self._cfg.correlated_bkg.signal_hist_name.format(
                    pt_min=f"{self._cfg.pt_min*10:.0f}",
                    pt_max=f"{self._cfg.pt_max*10:.0f}",
                    cent_min=self._cfg.cent_min,
                    cent_max=self._cfg.cent_max
                ),
                limits=self._cfg.mass_range,
                rebin=self._cfg.rebin
            )

            data_bkg_reference = DataHandler(
                data=bkg.file_norm,
                histoname=bkg.norm_hist_name.format(
                    pt_min=f"{self._cfg.pt_min*10:.0f}",
                    pt_max=f"{self._cfg.pt_max*10:.0f}",
                    cent_min=self._cfg.cent_min,
                    cent_max=self._cfg.cent_max
                ),
                limits=self._cfg.mass_range,
                rebin=self._cfg.rebin
            )

            return data_bkg_reference.get_norm() * \
                   bkg.br.pdg / bkg.br.simulations / \
                   (data_signal_reference.get_norm() * \
                   self._cfg.correlated_bkg.signal_br.pdg / \
                   self._cfg.correlated_bkg.signal_br.simulations)

        return None


if __name__ == "__main__":
    # Correlated background configuration
    corr_bkg_config = CorrelatedBackgroundConfig(
        fix_to_file=False,
        fix_to_mb=False,
        fix_with_br=True,
        file_name_for_fix=None,
        hist_name_for_fix="h_corr_bkg_over_dplus_signal_0_100",
        backgrounds=[
            CorrelatedBackground(
                name=r"$\mathrm{D^{+}}\rightarrow K^{-}\pi^{+}\pi^{+}$",
                file_norm="MC/xx/Projections/w_bdt/dplus_bkg.root",
                norm_hist_name='h_mass_{pt_min}_{pt_max}_cent_0_100',
                template_file="MC/xx/Projections/wo_bdt/dplus_corr_bkg_template.root",
                template_hist_name='h_mass_{pt_min}_{pt_max}',
                br=BRInfo(pdg=0.0938, simulations=0.4853)
            )
        ],
        signal_norm_file="MC/xx/Projections/w_bdt/dplus.root",
        signal_hist_name='h_mass_{pt_min}_{pt_max}_cent_0_100',
        signal_br=BRInfo(pdg=0.00269, simulations=0.4263)
    )

    # Fit configuration including correlated backgrounds
    config = FitConfig(
        pt_min=4.0,
        pt_max=6.0,
        cent_min=0,
        cent_max=100,
        mass_range=[1.75, 2.1],
        signal_pdfs=["gaussian", "gaussian"],
        bkg_pdfs=["chebpol2"],
        rebin=4,
        param_setup=[
            {"sigma": {
                "init": 0.005,
                "min": 0.0,
                "max": 0.05,
                "fix_to_config_value": False,
                "fix_to_file": False
            }},
            {"sigma": {
                "init": 0.005,
                "min": 0.0,
                "max": 0.05,
                "fix_to_config_value": False,
                "fix_to_file": False
            }},
        ],
        data_path="data/doublecb/Projections/projections.root",
        fix_dplus_sigma_to_ds=True,
        correlated_bkg=corr_bkg_config,
        draw_figures=True,
        draw_formats=["png", "pdf", "root"],
        output_dir="fitter/fit_results/"

    )

    # Run the FitHandler with correlated backgrounds
    handler = FitHandler(config)

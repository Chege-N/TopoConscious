"""
fMRI preprocessing: NIfTI loading, atlas parcellation, signal extraction.
"""
import numpy as np
import nibabel as nib
from nilearn import datasets, input_data, signal


class Preprocessor:
    """
    Loads a BOLD NIfTI file, parcellates with the chosen atlas,
    and returns a (n_volumes × n_regions) time-series matrix.
    """

    ATLAS_MAP = {
        "aal": ("fetch_atlas_aal", 90),
        "schaefer100": ("fetch_atlas_schaefer_2018", 100),
        "destrieux": ("fetch_atlas_destrieux_2009", 148),
    }

    def __init__(self, atlas: str = "aal", tr: float = 2.0,
                 low_pass: float = 0.1, high_pass: float = 0.01,
                 smoothing_fwhm: float = 6.0):
        self.atlas_name = atlas
        self.tr = tr
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.smoothing_fwhm = smoothing_fwhm
        self._atlas_img, self._labels = self._load_atlas()

    def _load_atlas(self):
        if self.atlas_name == "aal":
            atlas = datasets.fetch_atlas_aal()
            return atlas.maps, atlas.labels
        elif self.atlas_name == "schaefer100":
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
            return atlas.maps, atlas.labels
        else:
            raise ValueError(f"Unknown atlas: {self.atlas_name}")

    def load_and_extract(self, bold_path) -> np.ndarray:
        """
        Returns cleaned time series: shape (n_volumes, n_regions).
        """
        masker = input_data.NiftiLabelsMasker(
            labels_img=self._atlas_img,
            standardize=True,
            detrend=True,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.tr,
            smoothing_fwhm=self.smoothing_fwhm,
            verbose=0,
        )
        ts = masker.fit_transform(str(bold_path))
        return ts  # (n_vols, n_regions)

    @property
    def region_labels(self):
        return self._labels

"""
BdPy MRI package

This package is a part of BdPy
"""

from .fmriprep import FmriprepData, create_bdata_fmriprep
from .glm import make_paradigm
from .image import export_brain_image
from .load_epi import load_epi
from .load_mri import load_mri
from .roi import (
    add_hcp_rois,
    add_hcp_visual_cortex,
    add_roilabel,
    add_roimask,
    add_rois,
    get_roiflag,
    merge_rois,
)
from .spm import create_bdata_spm_domestic

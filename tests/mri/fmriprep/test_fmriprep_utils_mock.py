""" 
Utilities for mock fmriprep tests (builders, loaders, expected data).

All fixtures here are tailored to fmriprep version 1.2 filename patterns and
directory layout. Update the constants below if you add coverage for other
versions.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage

import bdpy


@dataclass
class RunConfig:
    """Configuration for one synthetic functional run."""

    session: str
    run_no: int
    n_vol: int
    tr: float


class MockBidsBuilder:
    """Builder for a minimal, synthetic BIDS/fmriprep 1.2 tree."""

    fmriprep_version: str = "1.2"
    subject: str = "sub-0840"
    sessions: Tuple[str, ...] = ("ses-01", "ses-02", "ses-anat")
    run_numbers: Tuple[int, ...] = (1, 2, 3)
    voxel_shape: Tuple[int, int, int] = (2, 2, 2)
    surface_vertices: Tuple[int, int] = (3, 5)
    session_tr: ClassVar[Dict[str, float]] = {"ses-01": 2.0, "ses-02": 1.8}

    def __init__(self) -> None:
        """Initialize an empty temporary mock BIDS tree."""
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.label_mapper_path = self.root / "label_mapper.tsv"
        self.data_map: Dict[str, np.ndarray] = {}
        self.confounds_map: Dict[str, pd.DataFrame] = {}
        self.events_map: Dict[str, pd.DataFrame] = {}
        self.bold_json_map: Dict[str, float] = {}
        self.expected_runs: Dict[str, List[dict]] = {}
        self.expected_subject_data: OrderedDict[str, OrderedDict[str, List[dict]]] = OrderedDict()

    def cleanup(self) -> None:
        """Remove the temporary dataset tree."""
        self._tmp.cleanup()

    def build(self) -> None:
        """Create the full mock dataset and label mapper."""
        self._write_label_mapper()
        self._create_dataset()

    def _write_label_mapper(self) -> None:
        self.label_mapper_path.write_text("face\t1\nscene\t2\n", encoding="ascii")

    def _create_dataset(self) -> None:
        for session in self.sessions:
            raw_func = self.root / self.subject / session / "func"
            prep_func = self.root / "derivatives" / "fmriprep" / "fmriprep" / self.subject / session / "func"
            raw_func.mkdir(parents=True, exist_ok=True)
            prep_func.mkdir(parents=True, exist_ok=True)
            if session == "ses-anat":
                continue

            run_list: List[dict] = []
            for run_no in self.run_numbers:
                config = RunConfig(
                    session=session,
                    run_no=run_no,
                    n_vol=self._volume_count(run_no),
                    tr=self.session_tr.get(session, 2.0),
                )
                run_list.append(self._create_run_files(prep_func=prep_func, raw_func=raw_func, config=config))
            self.expected_runs[session] = run_list

        session_items = OrderedDict()
        for session in self.sessions:
            if session == "ses-anat":
                continue
            if session in self.expected_runs:
                session_items[session] = self.expected_runs[session]
        self.expected_subject_data = OrderedDict({self.subject: session_items})

    def _confounds_dataframe(self, config: RunConfig) -> pd.DataFrame:
        base = self._confound_base(config)
        rows = []
        for idx in range(config.n_vol):
            row_base = base + idx
            row = {
                "X": row_base,
                "Y": row_base + 0.1,
                "Z": row_base + 0.2,
                "RotX": row_base + 0.3,
                "RotY": row_base + 0.4,
                "RotZ": row_base + 0.5,
                "global_signal": row_base + 1.01 * (1 + (idx % 2)),
                "white_matter": row_base + 2.02 * (1 + (idx % 3) * 0.1),
                "csf": row_base + 3.03,
                "dvars": row_base + 4.04,
                "std_dvars": row_base + 5.05,
                "framewise_displacement": np.nan if idx % 3 == 0 else row_base + 6.06,
            }
            for c_idx in range(6):
                row[f"a_comp_cor_0{c_idx}"] = row_base + 10 + c_idx
                row[f"t_comp_cor_0{c_idx}"] = row_base + 20 + c_idx
                row[f"cosine0{c_idx}"] = row_base + 30 + c_idx
            rows.append(row)
        return pd.DataFrame(rows)

    def _event_dataframe(self, config: RunConfig) -> pd.DataFrame:
        rows = []
        for idx in range(config.n_vol):
            onset = idx * config.tr
            rows.append(
                {
                    "onset": onset,
                    "duration": config.tr,
                    "trial_type": float(config.run_no),
                    "stimulus_name": "face" if idx % 2 == 0 else "scene",
                    "category_index": float(config.run_no * 10 + idx),
                    "image_index": float(config.run_no * 100 + idx),
                    "response_time": float(0.5 + 0.1 * idx),
                    "original_run_number": float(config.run_no),
                }
            )
        return pd.DataFrame(rows)

    def _create_run_files(self, prep_func: Path, raw_func: Path, config: RunConfig) -> dict:
        base = f"{self.subject}_{config.session}_task-mock_run-{config.run_no:02d}"
        filenames = {
            "volume_native": f"{base}_space-T1w_desc-preproc_bold.nii.gz",
            "volume_standard": f"{base}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "surf_native_left": f"{base}_space-fsnative_hemi-L.func.gii",
            "surf_native_right": f"{base}_space-fsnative_hemi-R.func.gii",
            "surf_standard_left": f"{base}_space-fsaverage_hemi-L.func.gii",
            "surf_standard_right": f"{base}_space-fsaverage_hemi-R.func.gii",
            "surf_standard_41k_left": f"{base}_space-fsaverage6_hemi-L.func.gii",
            "surf_standard_41k_right": f"{base}_space-fsaverage6_hemi-R.func.gii",
            "surf_standard_10k_left": f"{base}_space-fsaverage5_hemi-L.func.gii",
            "surf_standard_10k_right": f"{base}_space-fsaverage5_hemi-R.func.gii",
            "confounds": f"{base}_desc-confounds_regressors.tsv",
        }
        events_df = self._event_dataframe(config)
        conf_df = self._confounds_dataframe(config)
        bold_json = {"RepetitionTime": config.tr}

        events_path = raw_func / f"{base}_events.tsv"
        bold_json_path = raw_func / f"{base}_bold.json"
        confounds_path = prep_func / filenames["confounds"]
        events_df.to_csv(events_path, sep="\t", index=False)
        conf_df.to_csv(confounds_path, sep="\t", index=False)
        bold_json_path.write_text(json.dumps(bold_json), encoding="ascii")

        self.confounds_map[str(confounds_path)] = conf_df
        self.events_map[str(events_path)] = events_df
        self.bold_json_map[str(bold_json_path)] = config.tr
        native_volume = prep_func / filenames["volume_native"]
        standard_volume = prep_func / filenames["volume_standard"]
        self._register_image(native_volume, config)
        self._write_nifti(standard_volume, self.data_map[str(native_volume)])
        self._write_surface_pair(
            prep_func / filenames["surf_native_left"],
            prep_func / filenames["surf_native_right"],
            config=config,
            base_offset=0,
        )
        self._write_surface_pair(
            prep_func / filenames["surf_standard_left"],
            prep_func / filenames["surf_standard_right"],
            config=config,
            base_offset=100,
        )
        self._write_surface_pair(
            prep_func / filenames["surf_standard_41k_left"],
            prep_func / filenames["surf_standard_41k_right"],
            config=config,
            base_offset=200,
        )
        self._write_surface_pair(
            prep_func / filenames["surf_standard_10k_left"],
            prep_func / filenames["surf_standard_10k_right"],
            config=config,
            base_offset=300,
        )

        run_files = {
            "volume_native": os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["volume_native"]),
            "volume_standard": os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["volume_standard"]),
            "surface_native": (
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_native_left"]),
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_native_right"]),
            ),
            "surface_standard": (
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_left"]),
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_right"]),
            ),
            "surface_standard_41k": (
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_41k_left"]),
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_41k_right"]),
            ),
            "surface_standard_10k": (
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_10k_left"]),
                os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["surf_standard_10k_right"]),
            ),
            "confounds": os.path.join("derivatives", "fmriprep", "fmriprep", self.subject, config.session, "func", filenames["confounds"]),
        }
        return run_files

    def _register_image(self, path: Path, config: RunConfig) -> None:
        session_offset = 0 if config.session == "ses-01" else 1000
        base = session_offset + config.run_no * 100
        data = (
            np.arange(np.prod(self.voxel_shape) * config.n_vol, dtype=float)
            .reshape(*self.voxel_shape, config.n_vol, order="F")
            + base
        )
        self.data_map[str(path)] = data
        self._write_nifti(path, data)

    def _write_nifti(self, path: Path, data: np.ndarray) -> None:
        """Persist a small 4D NIfTI that nipy/nibabel can read."""
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
        nib.save(img, path)

    def _write_surface_pair(self, left_path: Path, right_path: Path, config: RunConfig, base_offset: float) -> None:
        """Create paired GIFTI func files with predictable values."""
        session_offset = 0 if config.session == "ses-01" else 1000
        base = session_offset + config.run_no * 10 + base_offset
        left_vertices, right_vertices = self.surface_vertices
        self._write_gifti(left_path, n_vertex=left_vertices, n_time=config.n_vol, base=base)
        self._write_gifti(right_path, n_vertex=right_vertices, n_time=config.n_vol, base=base + 0.5)

    def _write_gifti(self, path: Path, n_vertex: int, n_time: int, base: float) -> None:
        arrays = []
        for t in range(n_time):
            arr = np.arange(n_vertex, dtype=np.float32) + base + t
            arrays.append(GiftiDataArray(arr))
        img = GiftiImage(darrays=arrays)
        nib.save(img, path)

    def _confound_base(self, config: RunConfig) -> float:
        session_offset = 0 if config.session == "ses-01" else 1000
        return session_offset + config.run_no * 100 + 50

    @staticmethod
    def _volume_count(run_no: int) -> int:
        """Return a modest but non-trivial volume count per run (>=10)."""
        return 9 + run_no


def build_expected_bdata_after_exclude(
    builder: MockBidsBuilder, label_mapper_dict: Dict[str, Dict[str, int]], exclude: Optional[Dict] = None
) -> bdpy.BData:
    """Reconstruct expected BData matching __create_bdata_fmriprep_subject output."""

    class _LabelMapper:
        def __init__(self, l2v_map: Dict[str, Dict[str, int]]):
            self._l2v_map = l2v_map
            self._v2l_map: Dict[str, Dict[int, str]] = {}

        def get_value(self, mkey: str, label: str) -> int:
            """Map a label to a value according to the label mapper, with error checking."""
            if mkey not in self._l2v_map:
                raise RuntimeError(f"{mkey} not found in label mapper")

            if label == "n/a":
                return np.nan

            if label not in self._l2v_map[mkey]:
                raise RuntimeError(f"{label} not found in label mapper for {mkey}")

            val = self._l2v_map[mkey][label]
            if mkey not in self._v2l_map:
                self._v2l_map[mkey] = {val: label}
            else:
                if label in self._v2l_map[mkey].values():
                    label_exist = self._v2l_map[mkey][val]
                    if label != label_exist:
                        raise RuntimeError("Invalid label-value mapping (possibly non-unique values in label mapper)")
                self._v2l_map[mkey].update({val: label})
            return val

        def dump(self) -> Dict[str, Dict[int, str]]:
            return self._v2l_map

    act_label_map = _LabelMapper(label_mapper_dict)

    exclude = exclude or {}
    include_sessions: OrderedDict[str, List[dict]] = OrderedDict()

    # Mirror the exclusion logic in create_bdata_fmriprep (subject/session/run/session-run)
    if "subject" in exclude and builder.subject in exclude["subject"]:
        include_sessions = OrderedDict()
    else:
        for s_idx, (ses, runs) in enumerate(builder.expected_runs.items()):
            if "session" in exclude and (s_idx + 1) in exclude["session"]:
                continue

            filtered_runs: List[dict] = []
            for r_idx, run in enumerate(runs):
                if "run" in exclude and (r_idx + 1) in exclude["run"]:
                    continue
                if "session/run" in exclude:
                    ex_runs = exclude["session/run"][s_idx] if s_idx < len(exclude["session/run"]) else None
                    if ex_runs is not None and (r_idx + 1) in ex_runs:
                        continue
                filtered_runs.append(run)

            if filtered_runs:
                include_sessions[ses] = filtered_runs
    voxel_list = []
    motion_list = []
    confounds_dict: Dict[str, Dict[int, np.ndarray]] = {}
    ses_label_list = []
    run_label_list = []
    block_label_list = []
    labels_list = []
    last_xyz = None
    last_ijk = None
    last_run = 0
    last_block = 0
    run_idx = 0

    for i, (ses, runs) in enumerate(include_sessions.items()):
        for j, run in enumerate(runs):
            run_idx += 1
            epi_path = builder.root / run["volume_native"]
            conf_path = builder.root / run["confounds"]
            base_name = Path(run["confounds"]).name.replace("_desc-confounds_regressors.tsv", "")
            event_path = builder.root / builder.subject / ses / "func" / f"{base_name}_events.tsv"
            bold_path = builder.root / builder.subject / ses / "func" / f"{base_name}_bold.json"

            img_arr = builder.data_map[str(epi_path)]
            voxel = img_arr.reshape(-1, img_arr.shape[-1], order="F").T
            voxel_list.append(voxel)

            i_len, j_len, k_len, _ = img_arr.shape
            ijk = np.array(np.unravel_index(np.arange(i_len * j_len * k_len), (i_len, j_len, k_len), order="F"))
            ijk_b = np.vstack([ijk, np.ones((1, i_len * j_len * k_len))])
            affine = np.delete(np.delete(np.eye(5), 3, axis=0), 3, axis=1)
            xyz = (affine @ ijk_b)[:-1]
            last_xyz = xyz
            last_ijk = ijk

            conf_pd = builder.confounds_map[str(conf_path)]
            motion_cols = [c for c in conf_pd.columns if c in ["X", "Y", "Z", "RotX", "RotY", "RotZ"]]
            motion = np.hstack([np.c_[conf_pd[c]] for c in motion_cols])
            motion_list.append(motion)

            conf_keys = [c for c in conf_pd.columns if c not in motion_cols]
            for key in conf_keys:
                arr = np.c_[conf_pd[key]]
                confounds_dict.setdefault(key, {})[run_idx] = arr

            events = builder.events_map[str(event_path)]
            tr = builder.bold_json_map[str(bold_path)]
            blocks = []
            labels = []
            cols = events.columns.values
            cols = cols[~(cols == "onset")]
            cols = cols[~(cols == "duration")]
            for k, row in events.iterrows():
                nsmp = int(np.round(row["duration"] / tr))
                blocks.append(np.ones((nsmp, 1)) * (k + 1))
                label_vals = []
                for p in cols:
                    if p in label_mapper_dict:
                        v = act_label_map.get_value(p, row[p])
                    else:
                        v = row[p]
                    label_vals.append(v)
                label_vals = np.array([np.nan if x == "n/a" else float(x) for x in label_vals])
                labels.append(np.tile(label_vals, (nsmp, 1)))

            num_vol = voxel.shape[0]
            ses_label_list.append(np.ones((num_vol, 1)) * (i + 1))
            run_label_list.append(np.ones((num_vol, 1)) * (j + 1) + last_run)
            block_label_list.append(np.vstack(blocks) + last_block)
            labels_list.append(np.vstack(labels))
            last_block = block_label_list[-1][-1]
        last_run = run_label_list[-1][-1]

    braindata = np.vstack(voxel_list)
    motionparam = np.vstack(motion_list)
    ses_label = np.vstack(ses_label_list)
    run_label = np.vstack(run_label_list)
    block_label = np.vstack(block_label_list)
    labels_arr = np.vstack(labels_list)

    confounds = dict(confounds_dict)
    for c in list(confounds.keys()):
        conf_val_list = []
        for k in range(run_idx):
            if (k + 1) in confounds[c]:
                conf_val_list.append(confounds[c][k + 1])
            else:
                run_length = voxel_list[k].shape[0]
                nan_array = np.zeros([run_length, 1])
                nan_array[:, :] = np.nan
                conf_val_list.append(nan_array)
        confounds[c] = np.vstack(conf_val_list)

    expected_bdata = bdpy.BData()
    expected_bdata.add(braindata, "VoxelData")
    expected_bdata.add(ses_label, "Session")
    expected_bdata.add(run_label, "Run")
    expected_bdata.add(block_label, "Block")
    expected_bdata.add(labels_arr, "Label")
    expected_bdata.add(motionparam, "MotionParameter")

    # Unit selectors mirror the column order of the 6 rigid-body motion regressors
    # (translations then rotations) so tests can assert the same metadata as the
    # production pipeline without guessing column positions.
    motion_metadata = [
        ("MotionParameter_trans_x", [1, 0, 0, 0, 0, 0], "Motion parameter: x translation"),
        ("MotionParameter_trans_y", [0, 1, 0, 0, 0, 0], "Motion parameter: y translation"),
        ("MotionParameter_trans_z", [0, 0, 1, 0, 0, 0], "Motion parameter: z translation"),
        ("MotionParameter_rot_x", [0, 0, 0, 1, 0, 0], "Motion parameter: x rotation"),
        ("MotionParameter_rot_y", [0, 0, 0, 0, 1, 0], "Motion parameter: y rotation"),
        ("MotionParameter_rot_z", [0, 0, 0, 0, 0, 1], "Motion parameter: z rotation"),
    ]
    for key, basis, desc in motion_metadata:
        expected_bdata.add_metadata(key, basis, desc, where="MotionParameter")

    if confounds:
        default_confounds_keys = ["global_signal", "white_matter", "csf", "dvars", "std_dvars", "framewise_displacement"]
        confounds_key_desc = {
            "global_signal": {"key": "GlobalSignal", "desc": "Confounds: Average signal in brain mask"},
            "white_matter": {"key": "WhiteMatterSignal", "desc": "Confounds: Average signal in white matter"},
            "csf": {"key": "CSFSignal", "desc": "Confounds: Average signal in CSF"},
            "dvars": {"key": "DVARS", "desc": "Confounds: Original DVARS"},
            "std_dvars": {"key": "STD_DVARS", "desc": "Confounds: Standardized DVARS"},
            "framewise_displacement": {"key": "FramewiseDisplacement", "desc": "Confounds: Framewise displacement (bulk-head motion)"},
        }
        additional_confounds_series = [
            ("a_comp_cor", "aCompCor", "Confounds: Anatomical CompCor"),
            ("t_comp_cor", "tCompCor", "Confounds: Temporal CompCor"),
            ("cosine", "Cosine", "Confounds: Discrete cosine-basis regressors"),
        ]
        for target_key, target_key_name, target_key_desc in additional_confounds_series:
            default_confounds_keys.append(target_key)
            confounds_key_desc[target_key] = {"key": target_key_name, "desc": target_key_desc}
            ack_list = sorted([ack for ack in confounds.keys() if target_key in ack])
            for ack_i, ack in enumerate(ack_list):
                default_confounds_keys.append(ack)
                confounds_key_desc[ack] = {"key": f"{target_key_name}_{ack_i}", "desc": target_key_desc}

        confounds_array = np.hstack([confounds[dck] for dck in default_confounds_keys if dck in confounds])
        expected_bdata.add(confounds_array, "Confounds")
        cnf_p = 0
        for cnf in default_confounds_keys:
            if (cnf not in ["a_comp_cor", "t_comp_cor", "cosine"]) and (cnf not in confounds):
                continue
            cnf_colidx = np.zeros(confounds_array.shape[1])
            if cnf in ["a_comp_cor", "t_comp_cor", "cosine"]:
                ncol = sum([1 for k in confounds.keys() if cnf in k])
                for k in range(ncol):
                    cnf_colidx[cnf_p + k] = 1
            else:
                cnf_colidx[cnf_p] = 1
                cnf_p += 1
            expected_bdata.add_metadata(confounds_key_desc[cnf]["key"], cnf_colidx, confounds_key_desc[cnf]["desc"], where="Confounds")

    label_cols = ["trial_type", "stimulus_name", "category_index", "image_index", "response_time", "original_run_number"]
    for idx, col in enumerate(label_cols):
        expected_bdata.add_metadata(col, np.array([np.nan if i != idx else 1 for i in range(len(label_cols))], dtype=float), f"Label {col}", where="Label")

    if last_xyz is not None and last_ijk is not None:
        expected_bdata.add_metadata("voxel_x", last_xyz[0, :], "Voxel x coordinate", where="VoxelData")
        expected_bdata.add_metadata("voxel_y", last_xyz[1, :], "Voxel y coordinate", where="VoxelData")
        expected_bdata.add_metadata("voxel_z", last_xyz[2, :], "Voxel z coordinate", where="VoxelData")
        expected_bdata.add_metadata("voxel_i", last_ijk[0, :], "Voxel i index", where="VoxelData")
        expected_bdata.add_metadata("voxel_j", last_ijk[1, :], "Voxel j index", where="VoxelData")
        expected_bdata.add_metadata("voxel_k", last_ijk[2, :], "Voxel k index", where="VoxelData")

    vmap = act_label_map.dump()
    for k in vmap:
        expected_bdata.add_vmap(k, vmap[k])

    return expected_bdata

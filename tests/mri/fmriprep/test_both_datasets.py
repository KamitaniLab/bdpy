"""Dataset-agnostic tests for bdpy.mri.fmriprep, run against every fixture.

The mock and real-data suites exercise the same production code, but until now
they shared no test bodies: only ``create_bdata_fmriprep`` was ever driven by
the real fixture, so assumptions baked into the synthetic dataset (numeric
session names, every run having a BOLD file) went unchecked against real
fMRIPrep output.

This module holds the test bodies that assert properties which must hold for
*every* fMRIPrep dataset — the classes below call those properties invariants.
Each production class or function gets one such class; the concrete classes at
the bottom supply nothing but the location of a dataset and its expected
values. Adding one method to a class here therefore extends both the mock and
the real-data suites at once.

The test for belonging here is whether the assertion would still make sense
against a dataset nobody has seen yet. A test that never touches a dataset
(``BrainData(dtype="unsupported")`` raising, say) does not belong; it lives in
``test_mock_only.py`` with the other unit-level tests.

Cost note: ``create_bdata_fmriprep`` over a whole subject takes minutes on the
real fixture, so it is only reached here through the exclude-everything short
circuit. ``BrainData``, ``LabelMapper`` and ``create_bdata_singlesubject`` are
driven one run or one file at a time instead, which puts the whole real-data
run at roughly a minute and, unlike ``test_real_only.py``, needs no 2.5 GB
stored-expectation file — only the 478 MB dataset.
"""

from __future__ import annotations

import csv
import inspect
import unittest
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pytest

from ._support import RealDatasetMixin, fmriprep
from ._mock_fixtures import DATA_BUILDER, MockBidsBuilder, MockDatasetMixin

if TYPE_CHECKING:
    #: Type-checking only. The invariant classes below call ``self.assertEqual``
    #: and friends, but must not be ``unittest.TestCase`` subclasses at runtime:
    #: pytest collects every TestCase subclass regardless of its name, and would
    #: run these with the subclass-supplied attributes unset. The concrete
    #: classes at the bottom pick up ``TestCase`` from the dataset mixins.
    _AssertsOnly = unittest.TestCase
else:
    _AssertsOnly = object


#: Keys that ``FmriprepData.__parse_session`` must place on every run.
RUN_KEYS = (
    "volume_native",
    "volume_standard",
    "surface_native",
    "surface_standard",
    "surface_standard_41k",
    "surface_standard_10k",
    "confounds",
    "task_event_file",
    "bold_json",
)


def read_label_mapper_file(path: str) -> dict[str, int]:
    """Read a label mapper file into a label-to-value dict.

    ``create_bdata_fmriprep`` parses these files inline
    (bdpy/mri/fmriprep.py L244-258) rather than exposing a reusable helper, so
    the parsing is repeated here to reach ``LabelMapper`` directly.
    """
    delimiter = "," if Path(path).suffix == ".csv" else "\t"
    mapping: dict[str, int] = {}
    with open(path, "r") as f:
        for row in csv.reader(f, delimiter=delimiter):
            if not row:
                continue
            mapping[row[0]] = int(row[1])
    return mapping


class DatasetInvariants(_AssertsOnly):
    """Shared helpers for the invariant classes below.

    Deliberately not a ``unittest.TestCase`` and deliberately not named
    ``Test*``: pytest would otherwise collect this class directly and run it
    with the subclass-supplied attributes unset. The concrete classes pick up
    ``TestCase`` from ``MockDatasetMixin`` / ``RealDatasetMixin`` instead.
    """

    #: Production API this class exercises. Aggregated by ``ApiCoverageCheck``.
    covers: tuple[str, ...] = ()

    # --- supplied by the concrete subclasses ---------------------------------
    data_root: Path
    label_mapper: Optional[dict[str, str]]
    expected_subjects: list[str]
    expected_sessions: list[str]
    #: data_modes this dataset actually carries files for.
    supported_modes: tuple[str, ...] = ("volume_native",)
    fmriprep_dir: str = "derivatives/fmriprep"

    @property
    def prep_dir(self) -> str:
        """Path to the fmriprep output root inside the dataset."""
        return (self.data_root / self.fmriprep_dir / "fmriprep").as_posix()

    def make_fmriprep_data(self) -> Any:
        """Build a ``FmriprepData`` over this dataset."""
        return fmriprep.FmriprepData(
            self.data_root.as_posix(), fmriprep_dir=self.fmriprep_dir
        )

    def parsed_label_mapper(self) -> dict[str, dict[str, int]]:
        """Return this dataset's label mapper in the parsed dict-of-dicts form."""
        if self.label_mapper is None:
            return {}
        return {
            key: read_label_mapper_file(path)
            for key, path in self.label_mapper.items()
        }

    def first_run_with(self, data_mode: str) -> tuple[str, dict]:
        """Return the first (session, run) that carries files for ``data_mode``.

        Not every run has every modality: the real fixture ships run-01 as a
        bookkeeping placeholder with events and bold.json but no BOLD image.
        """
        data = self.make_fmriprep_data().data[self.expected_subjects[0]]
        for session, runs in data.items():
            for run in runs:
                value = run.get(data_mode)
                if value is None:
                    continue
                if isinstance(value, tuple) and any(v is None for v in value):
                    continue
                return session, run
        raise AssertionError(f"No run carries {data_mode} in {self.data_root}")


class FmriprepDataInvariants(DatasetInvariants):
    """Directory scanning must find the same structure in any dataset."""

    covers: tuple[str, ...] = (
        "FmriprepData.data",
        "FmriprepData.__parse_data",
        "FmriprepData.__parse_session",
        "FmriprepData.__get_subjects",
        "FmriprepData.__get_sessions",
        "FmriprepData.__get_task_event_files",
    )

    def test_configuration_fields_initialized(self) -> None:
        """Constructor arguments are stored verbatim."""
        instance = self.make_fmriprep_data()
        self.assertEqual(
            instance._FmriprepData__datapath, self.data_root.as_posix()
        )
        self.assertEqual(instance._FmriprepData__fmriprep_dir, self.fmriprep_dir)
        self.assertEqual(instance._FmriprepData__fmriprep_version, "1.2")

    def test_get_subjects(self) -> None:
        """Subject discovery returns exactly the dataset's subjects."""
        instance = self.make_fmriprep_data()
        subjects = instance._FmriprepData__get_subjects(self.prep_dir)
        self.assertEqual(subjects, self.expected_subjects)

    def test_get_sessions(self) -> None:
        """Session discovery returns exactly the dataset's sessions.

        Session names are not required to be numeric, and a session directory
        with no runs still counts as a session.
        """
        instance = self.make_fmriprep_data()
        sessions = instance._FmriprepData__get_sessions(
            self.prep_dir, self.expected_subjects[0]
        )
        self.assertEqual(sessions, self.expected_sessions)

    def test_parsed_data_is_keyed_by_subject_then_session(self) -> None:
        """`.data` maps every expected subject to a session mapping."""
        data = self.make_fmriprep_data().data
        self.assertEqual(list(data.keys()), self.expected_subjects)
        for subject in self.expected_subjects:
            for session in data[subject]:
                self.assertIn(session, self.expected_sessions)
                self.assertIsInstance(data[subject][session], list)

    def test_every_run_carries_all_run_keys(self) -> None:
        """Every parsed run exposes the full set of modality keys.

        Values may be ``None`` — the real fixture's run-01 has no BOLD image —
        but the keys themselves must always be present so callers can probe a
        modality without a KeyError.
        """
        data = self.make_fmriprep_data().data
        seen_any_run = False
        for subject_data in data.values():
            for runs in subject_data.values():
                for run in runs:
                    seen_any_run = True
                    for key in RUN_KEYS:
                        self.assertIn(key, run)
        self.assertTrue(seen_any_run, "dataset produced no runs at all")

    def test_task_event_and_bold_json_exist_for_every_run(self) -> None:
        """Every run points at an events file and a bold.json that both exist."""
        data = self.make_fmriprep_data().data
        for subject_data in data.values():
            for runs in subject_data.values():
                for run in runs:
                    for key in ("task_event_file", "bold_json"):
                        self.assertIsNotNone(run[key], f"{key} missing on {run}")
                        self.assertTrue(
                            (self.data_root / run[key]).exists(),
                            f"{key} does not exist: {run[key]}",
                        )


class LabelMapperInvariants(DatasetInvariants):
    """A dataset's label mapper must cover the labels its events use."""

    covers: tuple[str, ...] = ("LabelMapper.get_value", "LabelMapper.dump")

    def _event_labels(self) -> set[str]:
        """Collect every non-``n/a`` stimulus name used in the raw events."""
        labels: set[str] = set()
        for events in sorted(self.data_root.rglob("*_events.tsv")):
            if self.fmriprep_dir.split("/")[0] in events.parts:
                continue
            with events.open(newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    name = row.get("stimulus_name")
                    if name and name != "n/a":
                        labels.add(name)
        return labels

    def test_mapper_covers_every_label_used_in_events(self) -> None:
        """Every label appearing in the events files resolves to a value.

        A label present in the events but absent from the mapper surfaces as a
        bare ``KeyError`` deep inside ``create_bdata_fmriprep``; checking it
        here names the actual problem.
        """
        parsed = self.parsed_label_mapper()
        if not parsed:
            self.skipTest("dataset has no label mapper")
        mapper = fmriprep.LabelMapper(parsed)
        missing = sorted(
            label for label in self._event_labels() if label not in parsed["stimulus_name"]
        )
        self.assertEqual(missing, [], f"labels missing from the mapper: {missing}")
        for label in sorted(self._event_labels()):
            self.assertIsInstance(mapper.get_value("stimulus_name", label), int)

    def test_na_maps_to_nan(self) -> None:
        """The reserved ``n/a`` label resolves to NaN rather than raising."""
        parsed = self.parsed_label_mapper()
        if not parsed:
            self.skipTest("dataset has no label mapper")
        mapper = fmriprep.LabelMapper(parsed)
        self.assertTrue(np.isnan(mapper.get_value("stimulus_name", "n/a")))

    def test_dump_returns_the_inverse_of_queried_labels(self) -> None:
        """`dump` reports value-to-label pairs for the labels queried so far."""
        parsed = self.parsed_label_mapper()
        if not parsed:
            self.skipTest("dataset has no label mapper")
        mapper = fmriprep.LabelMapper(parsed)
        self.assertEqual(mapper.dump(), {})
        queried = sorted(self._event_labels())[:5]
        for label in queried:
            mapper.get_value("stimulus_name", label)
        dumped = mapper.dump()["stimulus_name"]
        self.assertEqual(sorted(dumped.values()), queried)
        for value, label in dumped.items():
            self.assertEqual(parsed["stimulus_name"][label], value)


class BrainDataInvariants(DatasetInvariants):
    """Loading one run must yield mutually consistent data and coordinates."""

    covers: tuple[str, ...] = (
        "BrainData.data",
        "BrainData.xyz",
        "BrainData.index",
        "BrainData.n_vertex",
        "BrainData.__load_volume",
        "BrainData.__load_surface",
        "BrainData.__load_surf_func_file",
    )

    def test_volume_data_xyz_and_index_agree_on_voxel_count(self) -> None:
        """Volume loading returns data, xyz and ijk over the same voxels."""
        _, run = self.first_run_with("volume_native")
        path = (self.data_root / run["volume_native"]).as_posix()
        brain = fmriprep.BrainData(path, dtype="volume")
        self.assertEqual(brain.data.ndim, 2)
        n_voxel = brain.data.shape[1]
        self.assertEqual(brain.xyz.shape, (3, n_voxel))
        self.assertEqual(brain.index.shape, (3, n_voxel))

    def test_surface_data_matches_hemisphere_vertex_counts(self) -> None:
        """Surface loading concatenates both hemispheres and reports their sizes."""
        if "surface_native" not in self.supported_modes:
            self.skipTest(
                "dataset ships no surface files "
                "(the real ds006319 fixture contains no *.gii)"
            )
        _, run = self.first_run_with("surface_native")
        left, right = run["surface_native"]
        brain = fmriprep.BrainData(
            ((self.data_root / left).as_posix(), (self.data_root / right).as_posix()),
            dtype="surface",
        )
        n_left, n_right = brain.n_vertex
        self.assertEqual(brain.data.shape[1], n_left + n_right)
        self.assertEqual(brain.index.shape, (1, n_left + n_right))
        with self.assertRaises(NotImplementedError):
            _ = brain.xyz


class SingleSubjectInvariants(DatasetInvariants):
    """One run through the subject-level builder must produce aligned columns."""

    covers: tuple[str, ...] = ("create_bdata_singlesubject",)

    def test_single_run_bdata_lines_up_with_its_source_image(self) -> None:
        """One run yields a BData whose rows and columns match the source.

        Rows: every per-sample column has as many rows as the brain data.
        Columns: the voxel count equals that of the run's own image, which
        avoids hard-coding a dataset-specific number.

        Deliberately a single test rather than one per assertion: this is the
        only expensive call the shared tests make on the real fixture
        (~40 s for one run), and splitting it would double that for no extra
        coverage.
        """
        session, run = self.first_run_with("volume_native")
        brain = fmriprep.BrainData(
            (self.data_root / run["volume_native"]).as_posix(), dtype="volume"
        )
        brain_data = fmriprep.create_bdata_singlesubject(
            subject_data=OrderedDict({session: [run]}),
            data_path=self.data_root.as_posix(),
            data_mode="volume_native",
            label_mapper=self.parsed_label_mapper(),
            with_confounds=True,
        )
        voxel_data = brain_data.get("VoxelData")
        n_sample = voxel_data.shape[0]
        self.assertGreater(n_sample, 0)
        self.assertEqual(voxel_data.shape[1], brain.data.shape[1])
        for key in ("Session", "Run", "Block", "Confounds"):
            self.assertEqual(
                brain_data.get(key).shape[0],
                n_sample,
                f"{key} row count does not match VoxelData",
            )
        for key in ("Session", "Run", "Block"):
            self.assertEqual(brain_data.get(key).shape[1], 1)


class CreateBdataInvariants(DatasetInvariants):
    """Top-level entry point invariants that cost nothing to check."""

    covers: tuple[str, ...] = ("create_bdata_fmriprep",)

    def test_excluding_every_subject_returns_empty_lists(self) -> None:
        """Excluding all subjects yields no output and reads no volumes.

        This returns in milliseconds even on the real fixture, so it also
        guards the short circuit: were the exclusion applied after loading,
        this test would take minutes instead.
        """
        bdata_list, data_labels_list = fmriprep.create_bdata_fmriprep(
            dpath=self.data_root.as_posix(),
            data_mode="volume_native",
            fmriprep_dir=self.fmriprep_dir,
            label_mapper=self.label_mapper,
            exclude={"subject": list(self.expected_subjects)},
            split_task_label=False,
            with_confounds=False,
            return_data_labels=True,
            return_list=True,
        )
        self.assertEqual(bdata_list, [])
        self.assertEqual(data_labels_list, [])


#: Every invariant class, in the order the concrete classes inherit them.
INVARIANT_CLASSES = (
    FmriprepDataInvariants,
    LabelMapperInvariants,
    BrainDataInvariants,
    SingleSubjectInvariants,
    CreateBdataInvariants,
)

#: Production API with no test, and why. Adding an entry requires a reason.
UNCOVERED_API: dict[str, str] = {}

#: Behaviours that no fixture reaches. Not API names, so they do not
#: participate in the completeness check below; they are the backlog.
UNCOVERED_BEHAVIOUR: dict[str, str] = {
    "create_bdata_fmriprep: split_task_label with multiple tasks": (
        "Both fixtures carry a single task, so the branch that returns more "
        "than one BData is never taken. Needs MockBidsBuilder to grow a second "
        "task, which forces the stored expectation files to be regenerated."
    ),
    "__create_bdata_fmriprep_subject: cut_duration < 0": (
        "The NotImplementedError path needs an events file whose duration "
        "exceeds the run length."
    ),
    "BrainData surface loading on real data": (
        "The ds006319 fixture ships no *.gii, so surface loading is exercised "
        "against synthetic data only."
    ),
}


def public_api_names() -> set[str]:
    """Enumerate the production API of ``bdpy.mri.fmriprep``.

    Name-mangled private methods are reported under their written name
    (``FmriprepData.__get_subjects``) so the ``covers`` declarations read the
    way the source does.
    """
    names: set[str] = set()
    for name, obj in vars(fmriprep).items():
        if name.startswith("__"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == fmriprep.__name__:
            names.add(name)
        elif inspect.isclass(obj) and obj.__module__ == fmriprep.__name__:
            prefix = f"_{obj.__name__}__"
            for attr, value in vars(obj).items():
                if attr.startswith("__") and attr.endswith("__"):
                    continue
                if not (inspect.isfunction(value) or isinstance(value, property)):
                    continue
                written = (
                    f"__{attr[len(prefix):]}" if attr.startswith(prefix) else attr
                )
                names.add(f"{obj.__name__}.{written}")
    return names


class ApiCoverageCheck(_AssertsOnly):
    """Fail when production code grows an API that no test claims.

    Dataset-independent, so the concrete real-data class below does not
    inherit it.
    """

    def test_every_public_api_is_claimed_or_excused(self) -> None:
        """Each production API is covered by a test class or listed as uncovered."""
        claimed: set[str] = set()
        for cls in INVARIANT_CLASSES:
            claimed.update(cls.covers)
        unclaimed = sorted(public_api_names() - claimed - set(UNCOVERED_API))
        self.assertEqual(
            unclaimed,
            [],
            "These APIs have no test and no entry in UNCOVERED_API. Add a test "
            f"to the matching invariant class, or record why not: {unclaimed}",
        )

    def test_covers_declarations_name_real_apis(self) -> None:
        """No ``covers`` entry refers to an API that no longer exists."""
        known = public_api_names()
        for cls in INVARIANT_CLASSES:
            stale = sorted(set(cls.covers) - known)
            self.assertEqual(
                stale, [], f"{cls.__name__}.covers names missing APIs: {stale}"
            )

    def test_uncovered_api_entries_name_real_apis(self) -> None:
        """No ``UNCOVERED_API`` entry refers to an API that no longer exists."""
        stale = sorted(set(UNCOVERED_API) - public_api_names())
        self.assertEqual(stale, [], f"UNCOVERED_API names missing APIs: {stale}")


class TestFmriprepInvariantsMock(
    FmriprepDataInvariants,
    LabelMapperInvariants,
    BrainDataInvariants,
    SingleSubjectInvariants,
    CreateBdataInvariants,
    ApiCoverageCheck,
    MockDatasetMixin,
):
    """Run the shared tests against the synthetic dataset."""

    expected_subjects = [MockBidsBuilder.subject]
    expected_sessions = list(MockBidsBuilder.sessions)
    supported_modes = (
        "volume_native",
        "surface_native",
        "surface_standard",
        "surface_standard_41k",
        "surface_standard_10k",
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Point the shared tests at the synthetic dataset."""
        super().setUpClass()
        cls.data_root = DATA_BUILDER.root


@pytest.mark.real_data
@pytest.mark.real_data_quick
class TestFmriprepInvariantsReal(
    FmriprepDataInvariants,
    LabelMapperInvariants,
    BrainDataInvariants,
    SingleSubjectInvariants,
    CreateBdataInvariants,
    RealDatasetMixin,
):
    """Run the shared tests against the real ds006319 fixture.

    ``ApiCoverageCheck`` is deliberately absent: it inspects the module rather
    than the dataset, so running it a second time would only slow the
    opt-in suite down.
    """

    #: The shared tests assert on the dataset itself, not on a stored result.
    requires_golden_master = False

    expected_subjects = ["sub-S1"]
    #: ses-SoundTest01 holds no runs; the fixture ships it as a placeholder for
    #: the session bookkeeping, and scanning must still report it.
    expected_sessions = ["ses-SoundTest01", "ses-SoundTest02"]
    #: The published fixture contains T1w volumes only.
    supported_modes = ("volume_native",)


if __name__ == "__main__":
    unittest.main()

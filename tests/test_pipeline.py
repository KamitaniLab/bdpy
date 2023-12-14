"""Tests for bdpy.pipeline."""


from unittest import TestCase, TestLoader, TextTestRunner
import sys

import numpy as np
import yaml
import hydra

from bdpy.pipeline.config import init_hydra_cfg


class TestPipeline(TestCase):
    """Tests for bdpy.pipeline."""

    def setUp(self):
        cfg = {
            "key_int": 0,
            "key_str": "value",
        }
        with open("/tmp/testconf.yaml", "w") as f:
            yaml.dump(cfg, f)

    def _rest_argv(self):
        if len(sys.argv) > 1:
            del sys.argv[1:]
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # hydra-core 1.0.6

    def test_config_init_hydra_cfg_default(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg.key_int, 0)
        self.assertEqual(cfg.key_str, "value")

    def test_config_init_hydra_cfg_override(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-o")
        sys.argv.append("key_int=1")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg.key_int, 1)
        self.assertEqual(cfg.key_str, "value")

        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-o")
        sys.argv.append("key_str=hoge")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg.key_int, 0)
        self.assertEqual(cfg.key_str, "hoge")

        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-o")
        sys.argv.append("key_str='hoge fuga'")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg.key_int, 0)
        self.assertEqual(cfg.key_str, "hoge fuga")

        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-o")
        sys.argv.append("key_int=1024")
        sys.argv.append("key_str=foo")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg.key_int, 1024)
        self.assertEqual(cfg.key_str, "foo")

    def test_config_init_hydra_cfg_run(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg._run_.name, "test_pipeline")

        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-a")
        sys.argv.append("overridden_analysis_name")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg._run_.name, "test_pipeline")

    def test_config_init_hydra_cfg_analysis(self):
        """Tests for bdpy.pipeline.config.init_hydra_cfg."""
        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg._analysis_name_, "test_pipeline")

        self._rest_argv()
        sys.argv.append("/tmp/testconf.yaml")
        sys.argv.append("-a")
        sys.argv.append("overridden_analysis_name")
        cfg = init_hydra_cfg()
        self.assertEqual(cfg._analysis_name_, "overridden_analysis_name")


if __name__ == "__main__":
    suite = TestLoader().loadTestsFromTestCase(TestPipeline)
    TextTestRunner(verbosity=2).run(suite)

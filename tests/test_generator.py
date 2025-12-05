# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for generator.py functions."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cudag.core.generator import run_generator


class TestRunGenerator:
    """Tests for run_generator function."""

    @pytest.fixture
    def mock_renderer(self) -> MagicMock:
        """Create a mock renderer."""
        renderer = MagicMock()
        renderer.screen_class.meta.return_value.size = (1000, 1000)
        return renderer

    @pytest.fixture
    def mock_task(self, mock_renderer: MagicMock) -> MagicMock:
        """Create a mock task."""
        task = MagicMock()
        task.task_type = "test-task"
        task.generate_samples.return_value = []
        task.generate_tests.return_value = []
        return task

    @pytest.fixture
    def temp_config(self) -> Path:
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
name_prefix: test-dataset
seed: 42
tasks:
  test-task: 10
splits:
  train: 0.8
test:
  count: 5
  tolerance: 10
output:
  image_format: png
  image_quality: 95
""")
            return Path(f.name)

    def test_check_script_invocation_called(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that check_script_invocation is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch("cudag.core.generator.check_script_invocation") as mock_check:
                    with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                        mock_builder.return_value.build.return_value = Path(tmpdir)
                        with patch.object(sys, "argv", ["generator.py", "--config", str(temp_config)]):
                            run_generator(mock_renderer, [mock_task], config_path=temp_config)

                mock_check.assert_called_once()
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_dataset_naming_with_researcher(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test dataset naming includes researcher name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Create .researcher file
                Path(".researcher").write_text("Name: testuser\n")

                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = Path(tmpdir)
                    with patch.object(sys, "argv", ["generator.py", "--config", str(temp_config)]):
                        run_generator(mock_renderer, [mock_task], config_path=temp_config)

                # Check that config.output_dir was set with researcher name
                call_kwargs = mock_builder.call_args[1]
                config = call_kwargs["config"]
                assert "testuser" in str(config.output_dir)
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_seed_override(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that --seed argument overrides config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = Path(tmpdir)
                    with patch.object(
                        sys, "argv",
                        ["generator.py", "--config", str(temp_config), "--seed", "123"]
                    ):
                        run_generator(mock_renderer, [mock_task], config_path=temp_config)

                call_kwargs = mock_builder.call_args[1]
                config = call_kwargs["config"]
                assert config.seed == 123
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_extra_args(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that extra arguments are parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                config_modifier_called = []

                def config_modifier(config, args):
                    config_modifier_called.append(args.debug)

                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = Path(tmpdir)
                    with patch.object(
                        sys, "argv",
                        ["generator.py", "--config", str(temp_config), "--debug"]
                    ):
                        run_generator(
                            mock_renderer,
                            [mock_task],
                            config_path=temp_config,
                            extra_args=[
                                ("--debug", {"action": "store_true"}),
                            ],
                            config_modifier=config_modifier,
                        )

                assert config_modifier_called == [True]
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_post_build_callback(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that post_build callback is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                post_build_calls = []

                def post_build(output_dir, renderer):
                    post_build_calls.append((output_dir, renderer))

                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = Path(tmpdir) / "output"
                    with patch.object(sys, "argv", ["generator.py", "--config", str(temp_config)]):
                        run_generator(
                            mock_renderer,
                            [mock_task],
                            config_path=temp_config,
                            post_build=post_build,
                        )

                assert len(post_build_calls) == 1
                assert post_build_calls[0][1] is mock_renderer
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_returns_output_dir(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that run_generator returns the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                expected_output = Path(tmpdir) / "datasets" / "test-output"

                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = expected_output
                    with patch.object(sys, "argv", ["generator.py", "--config", str(temp_config)]):
                        result = run_generator(mock_renderer, [mock_task], config_path=temp_config)

                assert result == expected_output
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

    def test_build_tests_called(
        self,
        mock_renderer: MagicMock,
        mock_task: MagicMock,
        temp_config: Path,
    ) -> None:
        """Test that build_tests is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                with patch("cudag.core.generator.DatasetBuilder") as mock_builder:
                    mock_builder.return_value.build.return_value = Path(tmpdir)
                    with patch.object(sys, "argv", ["generator.py", "--config", str(temp_config)]):
                        run_generator(mock_renderer, [mock_task], config_path=temp_config)

                mock_builder.return_value.build_tests.assert_called_once()
            finally:
                os.chdir(original_cwd)
                temp_config.unlink()

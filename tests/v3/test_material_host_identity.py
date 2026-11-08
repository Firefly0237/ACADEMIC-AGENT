"""Actual local process birth/exit observations; no Docker/PG/provider."""
from __future__ import annotations

import subprocess
import sys

from citequest.materials_v3.host_identity import (
    boot_identity,
    current_process_identity,
    process_identity,
    stopped_original_process,
)


def test_current_process_identity_is_not_treated_as_stopped():
    value = current_process_identity()
    assert value["exists"] and value["running"] and value["birth"]
    assert stopped_original_process(value) is False


def test_same_PID_with_different_birth_cannot_be_the_original_process():
    value = current_process_identity()
    value["birth"] = "0"
    assert stopped_original_process(value) is True


def test_real_child_termination_is_observed_without_PID_only_inference():
    child = subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        value = process_identity(child.pid)
        assert value["running"] and stopped_original_process(value) is False
        child.stdin.close()
        assert child.wait(timeout=5) == 0
        assert stopped_original_process(value) is True
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_boot_identity_is_stable_across_two_actual_observations():
    assert boot_identity() == boot_identity()

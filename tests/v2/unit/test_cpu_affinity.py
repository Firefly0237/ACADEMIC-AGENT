"""Real SMT topology regression and refusal to expand the allowed CPU mask."""
import pytest

from citequest.materials_v2.cpu_affinity import CpuSet, select_affinity, system_cpu_sets
from citequest.materials_v2.text import MaterialError


def test_two_threads_on_same_core_are_not_preferred_over_distinct_cores():
    topology = [CpuSet(0, 0, 0, 1), CpuSet(0, 1, 0, 1), CpuSet(0, 2, 2, 1), CpuSet(0, 3, 2, 1), CpuSet(0, 4, 4, 0)]
    cpus, receipt = select_affinity([0, 1, 2, 3, 4], 2, topology)
    assert cpus == [0, 2] and receipt["distinct_physical_cores"] == 2
    assert receipt["logical_cpu_limit_unchanged"] == 2


def test_restricted_allowed_mask_and_single_core_siblings_remain_hard_limits():
    topology = [CpuSet(0, 0, 0, 1), CpuSet(0, 1, 0, 1), CpuSet(0, 2, 2, 1)]
    assert select_affinity([0, 1], 2, topology)[0] == [0, 1]
    assert select_affinity([1, 2], 2, topology)[0] == [1, 2]
    with pytest.raises(MaterialError, match="INSUFFICIENT"):
        select_affinity([0], 2, topology)


def test_parked_or_exclusively_allocated_foreign_cpu_is_not_selected():
    topology = [CpuSet(0, 0, 0, 1, flags=1), CpuSet(0, 1, 1, 1, flags=2), CpuSet(0, 2, 2, 1), CpuSet(0, 3, 3, 1, flags=6)]
    assert select_affinity([0, 1, 2, 3], 2, topology)[0] == [2, 3]


@pytest.mark.parametrize("allowed,count", [([0, 0], 2), ([0, True], 2), ([-1, 0], 2), ([0], True)])
def test_invalid_operator_mask_is_rejected(allowed, count):
    with pytest.raises(MaterialError):
        select_affinity(allowed, count, [CpuSet(0, 0, 0, 1)])


def test_actual_os_topology_selection_stays_in_process_mask():
    import psutil

    allowed = psutil.Process().cpu_affinity()
    cpus, receipt = select_affinity(allowed, 2, system_cpu_sets())
    assert len(set(cpus)) == 2 and set(cpus) <= set(allowed)
    assert receipt["logical_cpu_limit_unchanged"] == 2

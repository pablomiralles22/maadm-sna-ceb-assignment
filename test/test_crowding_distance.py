import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import itertools

from nsga_utils import CrowdingDistanceUtils

EPS = 1e-5


def test_calculate_crowding_distances_single_individual():
    fitness_values = [(1, 2)]
    assert CrowdingDistanceUtils.calculate(fitness_values) == [float("inf")]


def test_calculate_crowding_distances_two_individuals():
    fitness_values = [(1, 2), (3, 4)]
    assert CrowdingDistanceUtils.calculate(fitness_values) == [
        float("inf"),
        float("inf"),
    ]


def test_calculate_crowding_distances_three_individuals():
    fitness_values = [(1, 2), (3, 4), (5, 6)]

    result = CrowdingDistanceUtils.calculate(fitness_values)
    expected_result = [float("inf"), 2.0, float("inf")]

    assert result == pytest.approx(expected_result, EPS)


def test_calculate_crowding_distances_multiple_objectives():
    fitness_values = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

    expected_result = [float("inf"), 3.0, float("inf")]

    for input_perm, expected_result_perm in zip(
        itertools.permutations(fitness_values), itertools.permutations(expected_result)
    ):
        result = CrowdingDistanceUtils.calculate(input_perm)
        assert result == pytest.approx(expected_result_perm, EPS)


def test_calculate_crowding_distances_non_strict_domination():
    fitness_values = [(0, 0), (1, 1), (1, 1), (3, 4)]

    result = CrowdingDistanceUtils.calculate(fitness_values)
    expected_result = [
        float("inf"),
        1.0 / 3.0 + 1.0 / 4.0,
        2.0 / 3.0 + 3.0 / 4.0,
        float("inf"),
    ]

    assert result == pytest.approx(expected_result, EPS)


if __name__ == "__main__":
    pytest.main()

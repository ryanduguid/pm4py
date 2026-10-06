import importlib.util
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.optimize import OptimizeResult

from pm4py.util.lp import solver
from pm4py.util.lp.variants import scipy_solver


class ScipySolverIntegralityTest(unittest.TestCase):
    def test_integer_lower_bound_through_public_solver(self):
        for integrality in (1, [1], np.array([1])):
            with self.subTest(integrality=integrality):
                result = solver.apply(
                    [1], [[-1]], [-1.25], None, None,
                    variant=solver.SCIPY,
                    parameters={"integrality": integrality},
                )
                self.assertIsInstance(result, OptimizeResult)
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.x[0], 2)
                self.assertAlmostEqual(result.fun, 2)

    def test_integer_infeasible_equality(self):
        result = scipy_solver.apply(
            [1], None, None, [[1]], [0.5],
            parameters={"integrality": 1},
        )
        self.assertFalse(result.success)
        self.assertIsNone(solver.get_points_from_sol(result, variant=solver.SCIPY))
        self.assertIsNone(solver.get_prim_obj_from_sol(result, variant=solver.SCIPY))

    def test_mixed_domains_keep_the_continuous_coordinate(self):
        result = scipy_solver.apply(
            [0, 1], [[0, -1]], [-1.25], [[1, 0]], [0.5],
            parameters={"integrality": [0, 1]},
        )
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [0.5, 2])
        self.assertAlmostEqual(result.fun, 2)
        self.assertEqual(
            solver.get_points_from_sol(result, variant=solver.SCIPY), [0.5, 2.0]
        )

    def test_fractional_solution_through_public_getters(self):
        for coordinate in (0.25, -0.25):
            with self.subTest(coordinate=coordinate):
                result = solver.apply(
                    [1], None, None, [[1]], [coordinate],
                    variant=solver.SCIPY,
                    parameters={"method": "highs", "bounds": [(None, None)]},
                )
                self.assertTrue(result.success)
                objective = solver.get_prim_obj_from_sol(result, variant=solver.SCIPY)
                points = solver.get_points_from_sol(result, variant=solver.SCIPY)
                self.assertEqual(objective, coordinate)
                self.assertEqual(points, [coordinate])
                self.assertIs(type(objective), float)
                self.assertIs(type(points[0]), float)
                self.assertEqual(scipy_solver.get_prim_obj_from_sol(result), objective)
                self.assertEqual(scipy_solver.get_points_from_sol(result), points)

    def test_integer_solution_keeps_fractional_objective(self):
        result = solver.apply(
            [0.25], [[-1]], [-1.25], None, None,
            variant=solver.SCIPY, parameters={"integrality": [1]},
        )
        self.assertTrue(result.success)
        self.assertEqual(solver.get_points_from_sol(result, variant=solver.SCIPY), [2.0])
        self.assertEqual(solver.get_prim_obj_from_sol(result, variant=solver.SCIPY), 0.5)

    def test_unsuccessful_partial_results_are_not_solutions(self):
        for status in range(5):
            with self.subTest(status=status):
                result = OptimizeResult(
                    success=False, status=status, x=np.array([1.25]), fun=1.25,
                )
                self.assertIsNone(scipy_solver.get_points_from_sol(result))
                self.assertIsNone(scipy_solver.get_prim_obj_from_sol(result))
                self.assertIsNone(solver.get_points_from_sol(result, variant=solver.SCIPY))
                self.assertIsNone(solver.get_prim_obj_from_sol(result, variant=solver.SCIPY))
                np.testing.assert_array_equal(result.x, [1.25])
                self.assertEqual(result.fun, 1.25)

    def test_incomplete_results_and_independent_fields(self):
        for result, points, objective in (
            (None, None, None),
            (OptimizeResult(x=np.array([0.25]), fun=0.25), None, None),
            (OptimizeResult(success=True), None, None),
            (OptimizeResult(success=True, x=None, fun=None), None, None),
            (OptimizeResult(success=True, x=np.array([0.25])), [0.25], None),
            (OptimizeResult(success=True, fun=0.25), None, 0.25),
            (OptimizeResult(success=True, x=np.array([]), fun=0.0), [], 0.0),
        ):
            with self.subTest(result=result):
                self.assertEqual(solver.get_points_from_sol(result, variant=solver.SCIPY), points)
                self.assertEqual(solver.get_prim_obj_from_sol(result, variant=solver.SCIPY), objective)

    def test_zero_solution_and_independent_point_lists(self):
        result = OptimizeResult(success=True, x=np.array([0.0]), fun=0.0)
        parameters = {"caller_option": "preserved"}
        first = solver.get_points_from_sol(result, variant=solver.SCIPY, parameters=parameters)
        second = solver.get_points_from_sol(result, variant=solver.SCIPY, parameters=parameters)
        self.assertEqual(first, [0.0])
        self.assertEqual(solver.get_prim_obj_from_sol(result, variant=solver.SCIPY), 0.0)
        self.assertIsNot(first, second)
        first[0] = 7
        self.assertEqual(second, [0.0])
        np.testing.assert_array_equal(result.x, [0.0])
        self.assertEqual(parameters, {"caller_option": "preserved"})

    def test_marking_equation_rounds_counts_but_uses_raw_heuristic(self):
        from pm4py.algo.analysis.marking_equation.variants import classic

        equation = classic.MarkingEquationSolver.__new__(classic.MarkingEquationSolver)
        equation.c = [10, 1]
        equation.inv_indices = {0: "A", 1: "B"}
        with mock.patch.object(solver, "DEFAULT_LP_SOLVER_VARIANT", solver.SCIPY):
            h, points = equation.solve_given_components(
                equation.c, None, None, np.eye(2), [0.6, 0.4],
            )
        self.assertEqual(h, 6)
        self.assertEqual(points, [1, 0])
        self.assertEqual(equation.get_activated_transitions(points), ["A"])

    def test_extended_marking_equation_rounds_before_aggregation(self):
        from pm4py.algo.analysis.extended_marking_equation.variants import classic

        equation = classic.ExtendedMarkingEquationSolver.__new__(
            classic.ExtendedMarkingEquationSolver
        )
        equation.sync_net = SimpleNamespace(transitions=["A"])
        equation.x = [[0], [1]]
        equation.y = []
        equation.c1 = [10, 1]
        equation.inv_indices = {0: "A"}
        for coordinate, expected_count, expected_h in ((0.4, 0, 4), (0.6, 2, 6)):
            with self.subTest(coordinate=coordinate), mock.patch.object(
                equation, "get_components",
                return_value=(equation.c1, None, None, np.eye(2), [coordinate] * 2),
            ):
                h, points = equation.solve(variant=solver.SCIPY)
                self.assertEqual(h, expected_h)
                self.assertEqual(points, [expected_count])
                self.assertEqual(
                    equation.get_activated_transitions(points), ["A"] * expected_count
                )

    @staticmethod
    def _marking_equations():
        from pm4py.algo.analysis.marking_equation.variants import classic
        from pm4py.algo.analysis.extended_marking_equation.variants import (
            classic as extended,
        )

        equation = classic.MarkingEquationSolver.__new__(classic.MarkingEquationSolver)
        equation.c = [10, 1]
        extended_equation = extended.ExtendedMarkingEquationSolver.__new__(
            extended.ExtendedMarkingEquationSolver
        )
        extended_equation.sync_net = SimpleNamespace(transitions=["A"])
        extended_equation.x = [[0], [1]]
        extended_equation.y = []
        extended_equation.c1 = equation.c
        components = (equation.c, None, None, np.eye(2), [0.6, 0.4])
        return equation, extended_equation, components

    def test_marking_equations_preserve_other_backend_conversion(self):
        equation, extended_equation, components = self._marking_equations()
        with (
            mock.patch.object(solver, "DEFAULT_LP_SOLVER_VARIANT", solver.PULP),
            mock.patch.object(solver, "apply"),
            mock.patch.object(solver, "get_points_from_sol", return_value=[0.6, 0.4]),
            mock.patch.object(extended_equation, "get_components", return_value=components),
        ):
            self.assertEqual(equation.solve_given_components(*components), (6, [0, 0]))
            self.assertEqual(extended_equation.solve(variant=solver.PULP), (6, [1]))

    def test_marking_equations_reject_unsuccessful_partial_results(self):
        equation, extended_equation, components = self._marking_equations()
        result = OptimizeResult(success=False, x=np.array([0.6, 0.4]), fun=6.4)
        with (
            mock.patch.object(solver, "DEFAULT_LP_SOLVER_VARIANT", solver.SCIPY),
            mock.patch.object(solver, "apply", return_value=result),
            mock.patch.object(extended_equation, "get_components", return_value=components),
        ):
            self.assertEqual(equation.solve_given_components(*components), (None, None))
            self.assertEqual(extended_equation.solve(variant=solver.SCIPY), (None, None))

    def test_woflan_preserves_scipy_rounding_and_other_backend_values(self):
        from pm4py.algo.analysis.woflan.place_invariants import utility

        result = OptimizeResult(success=True, x=np.array([1.0, 0.6]), fun=0.0)
        for variant, expected in ((solver.SCIPY, 1.0), (solver.PULP, 0.6)):
            with (
                self.subTest(variant=variant),
                mock.patch.object(
                    utility.importlib.util, "find_spec",
                    return_value=None if variant == solver.SCIPY else object(),
                ),
                mock.patch.object(utility.constants, "SHOW_INTERNAL_WARNINGS", False),
                mock.patch.object(solver, "apply", return_value=result) as apply,
                mock.patch.object(solver, "get_prim_obj_from_sol", return_value=0.0),
            ):
                if variant == solver.SCIPY:
                    basis = utility.transform_basis([np.array([[2.0]])], style="uniform")
                else:
                    with mock.patch.object(solver, "get_points_from_sol", return_value=[1.0, 0.6]):
                        basis = utility.transform_basis([np.array([[2.0]])], style="uniform")
                self.assertEqual(apply.call_args.kwargs["variant"], variant)
                np.testing.assert_array_equal(basis, [[[expected]]])

    def test_alignment_heuristic_uses_raw_values_and_failed_result_fallback(self):
        from pm4py.objects.petri_net.utils import align_utils

        compute = getattr(align_utils, "__compute_exact_heuristic_new_version")
        incidence = SimpleNamespace(encode_marking=lambda marking: [0])
        net = SimpleNamespace(transitions=["A"])
        arguments = (net, [[1]], None, None, [1], incidence, None, [0.25], solver.SCIPY)
        for success in (True, False):
            result = OptimizeResult(success=success, x=np.array([0.25]), fun=0.25)
            with self.subTest(success=success), mock.patch.object(
                solver, "apply", return_value=result
            ):
                h, points = compute(*arguments)
                self.assertEqual(h, 0.25 if success else align_utils.sys.maxsize)
                self.assertEqual(points, [0.25] if success else [0.0])

    def test_semi_continuous_and_semi_integer_domains(self):
        for flag, aub, bub, expected in (
            (2, None, None, 0),
            (3, [[-1]], [-2.5], 3),
        ):
            with self.subTest(flag=flag):
                result = scipy_solver.apply(
                    [1], aub, bub, None, None,
                    parameters={"integrality": [flag], "bounds": [(2, 5)]},
                )
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.x[0], expected)

    def test_integer_tail_cost_preserves_coordinate_rounding(self):
        from pm4py.algo.conformance.alignments.petri_net.variants import (
            approx_fixed_horizon,
        )
        from tests.approx_alignment_test import sequence_net

        net, im, fm = sequence_net("A")
        costs = {transition: 10_000_000 for transition in net.transitions}
        result = OptimizeResult(success=True, x=np.array([1.0000001]), fun=10_000_001.0)
        with mock.patch.object(solver, "DEFAULT_LP_SOLVER_VARIANT", solver.SCIPY):
            self.assertEqual(
                approx_fixed_horizon._integer_tail_cost(net, im, fm, costs), 10_000_000
            )
            with mock.patch.object(solver, "apply", return_value=result):
                self.assertEqual(
                    approx_fixed_horizon._integer_tail_cost(net, im, fm, costs), 10_000_000
                )
        with (
            mock.patch.object(solver, "DEFAULT_LP_SOLVER_VARIANT", solver.PULP),
            mock.patch.object(solver, "apply"),
            mock.patch.object(solver, "get_points_from_sol", return_value=[1.0000001]),
        ):
            self.assertEqual(
                approx_fixed_horizon._integer_tail_cost(net, im, fm, costs), 10_000_001
            )

    def test_scalar_and_short_integrality_broadcast(self):
        for integrality in (1, [1], np.array([1])):
            with self.subTest(integrality=integrality):
                result = scipy_solver.apply(
                    [1, 1], [[-1, 0], [0, -1]], [-1.25, -2.25],
                    None, None, parameters={"integrality": integrality},
                )
                self.assertTrue(result.success)
                np.testing.assert_allclose(result.x, [2, 3])

    def test_continuous_default_and_parameter_preservation(self):
        for integrality in (None, 0, [0], np.array([0])):
            with self.subTest(integrality=integrality):
                parameters = {"integrality": integrality}
                with mock.patch.object(
                    scipy_solver, "linprog", wraps=scipy_solver.linprog
                ) as linprog:
                    result = scipy_solver.apply(
                        [1], [[-1]], [-1.25], None, None,
                        parameters=parameters,
                    )
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.x[0], 1.25)
                self.assertEqual(
                    linprog.call_args.kwargs["method"], "revised simplex"
                )
                self.assertEqual(list(parameters), ["integrality"])
                self.assertIs(parameters["integrality"], integrality)

    def test_explicit_continuous_methods(self):
        for method in ("revised simplex", "highs-ds", "highs-ipm"):
            with self.subTest(method=method):
                result = scipy_solver.apply(
                    [1], [[-1]], [-1.25], None, None,
                    parameters={"integrality": [0], "method": method},
                )
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.x[0], 1.25)

    def test_case_insensitive_highs_and_enum_parameters(self):
        class Keys(Enum):
            INTEGRALITY = "integrality"
            METHOD = "method"
            BOUNDS = "bounds"

        class Methods(Enum):
            HIGHS = "HiGhS"

        parameters = {
            Keys.INTEGRALITY: [1],
            Keys.METHOD: Methods.HIGHS,
            Keys.BOUNDS: [(0, 4)],
        }
        original = parameters.copy()
        result = scipy_solver.apply(
            [1], [[-1]], [-1.25], None, None, parameters=parameters,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[0], 2)
        self.assertEqual(parameters, original)

    def test_incompatible_integer_methods_fail_before_dispatch(self):
        for method in (
            "revised simplex", "simplex", "interior-point", "highs-ds",
            "highs-ipm", "unknown", None, 7,
        ):
            with self.subTest(method=method), mock.patch.object(
                scipy_solver, "linprog"
            ) as linprog:
                with self.assertRaisesRegex(ValueError, "method='highs'"):
                    scipy_solver.apply(
                        [1], [[-1]], [-1.25], None, None,
                        parameters={"integrality": [1], "method": method},
                    )
                linprog.assert_not_called()

    def test_explicit_none_continuous_method_is_forwarded(self):
        expected = OptimizeResult(success=False)
        with mock.patch.object(
            scipy_solver, "linprog", return_value=expected
        ) as linprog:
            result = scipy_solver.apply(
                [1], None, None, None, None, parameters={"method": None},
            )
        self.assertIs(result, expected)
        self.assertIsNone(linprog.call_args.kwargs["method"])

    def test_invalid_nonzero_shape_is_validated_by_scipy(self):
        integrality = [1, 0, 1]
        with mock.patch.object(
            scipy_solver, "linprog", wraps=scipy_solver.linprog
        ) as linprog:
            with self.assertRaises(ValueError):
                scipy_solver.apply(
                    [1, 1], None, None, None, None,
                    parameters={"integrality": integrality},
                )
        self.assertIs(linprog.call_args.kwargs["integrality"], integrality)
        self.assertEqual(linprog.call_args.kwargs["method"], "highs")

    def test_ilp_discovery_uses_highs_and_replays_training_traces(self):
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        from pm4py.algo.discovery.ilp.variants import classic
        from pm4py.objects.log.obj import EventLog
        from tests.approx_alignment_test import trace

        log = EventLog([trace("ABC"), trace("AC")])
        with mock.patch.object(
            scipy_solver, "linprog", wraps=scipy_solver.linprog
        ) as linprog:
            net, im, fm = classic.apply(
                log, parameters={"show_progress_bar": False}
            )
        self.assertGreater(linprog.call_count, 0)
        for call in linprog.call_args_list:
            self.assertTrue(np.any(call.kwargs["integrality"]))
            self.assertEqual(call.kwargs["method"], "highs")
        replay = token_replay.apply(
            log, net, im, fm, parameters={"show_progress_bar": False}
        )
        self.assertTrue(all(result["trace_is_fit"] for result in replay))

    def test_murata_scipy_reduction_preserves_trace_fit(self):
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        from pm4py.objects.log.obj import EventLog
        from pm4py.objects.petri_net.obj import PetriNet
        from pm4py.objects.petri_net.utils import murata, petri_utils
        from tests.approx_alignment_test import sequence_net, trace

        net, im, fm = sequence_net("AB")
        duplicate = PetriNet.Place("duplicate")
        net.places.add(duplicate)
        transitions = {transition.label: transition for transition in net.transitions}
        petri_utils.add_arc_from_to(transitions["A"], duplicate, net)
        petri_utils.add_arc_from_to(duplicate, transitions["B"], net)
        find_spec = importlib.util.find_spec
        with mock.patch.object(
            murata.importlib.util, "find_spec",
            side_effect=lambda name: None if name == "pulp" else find_spec(name),
        ), mock.patch.object(
            scipy_solver, "linprog", wraps=scipy_solver.linprog
        ) as linprog:
            net, im, fm = murata.apply_reduction(net, im, fm)
        self.assertEqual(len(net.places), 3)
        self.assertEqual(len(net.transitions), 2)
        self.assertGreater(linprog.call_count, 0)
        for call in linprog.call_args_list:
            self.assertEqual(call.kwargs["method"], "highs")
        replay = token_replay.apply(
            EventLog([trace("AB")]), net, im, fm,
            parameters={"show_progress_bar": False},
        )
        self.assertTrue(replay[0]["trace_is_fit"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from enum import Enum
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

    def test_mixed_domains_keep_the_continuous_coordinate(self):
        result = scipy_solver.apply(
            [0, 1], [[0, -1]], [-1.25], [[1, 0]], [0.5],
            parameters={"integrality": [0, 1]},
        )
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [0.5, 2])
        self.assertAlmostEqual(result.fun, 2)

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

import importlib.util
import sys
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from pm4py.util.lp import solver


class LpSolverTest(unittest.TestCase):
    def test_backend_selection(self):
        self.assertEqual(solver.SCIPY, solver.DEFAULT_LP_SOLVER_VARIANT)
        if not importlib.util.find_spec("pulp"):
            self.assertNotIn(solver.PULP, solver.VERSIONS_APPLY)
            return

        import pulp
        from pm4py.util.lp.variants import pulp_solver

        expected = "pulp_solver_v4" if hasattr(pulp, "LpSolveStatus") else "pulp_solver"
        self.assertEqual(expected, pulp_solver.__name__.rsplit(".", 1)[-1])
        self.assertIs(pulp_solver.apply, solver.VERSIONS_APPLY[solver.PULP])
        if hasattr(pulp, "LpSolveStatus"):
            self.assertNotIn("pm4py.util.lp.variants.pulp_solver", sys.modules)

    def _solve(self, c, Aub, bub, Aeq=None, beq=None, parameters=None):
        if solver.PULP not in solver.VERSIONS_APPLY:
            self.skipTest("PuLP is not installed")
        return solver.apply(c, Aub, bub, Aeq, beq, parameters, variant=solver.PULP)

    def _points(self, sol, parameters=None):
        return solver.get_points_from_sol(sol, parameters, variant=solver.PULP)

    def _objective(self, sol):
        return solver.get_prim_obj_from_sol(sol, variant=solver.PULP)

    def _v4_backend(self):
        if solver.PULP not in solver.VERSIONS_APPLY:
            self.skipTest("PuLP is not installed")
        from pm4py.util.lp.variants import pulp_solver
        if not hasattr(pulp_solver, "LpSolveStatus"):
            self.skipTest("Requires PuLP 4")
        return pulp_solver

    def test_continuous_and_integer_optima(self):
        for require_ilp, expected in ((False, 1.5), (True, 2.0)):
            with self.subTest(require_ilp=require_ilp):
                sol = self._solve(
                    [1, 2], [[-1, -1]], [-1.5],
                    parameters={"require_ilp": require_ilp, "bounds": [(0, None)] * 2},
                )
                self.assertAlmostEqual(expected, self._objective(sol))
                self.assertEqual([expected, 0.0], self._points(sol))

    def test_mixed_integrality_and_equalities(self):
        sol = self._solve(
            [1, 1], [[-1, -1]], [-1.5], [[1, 0]], [1],
            parameters={
                "require_ilp": True,
                "integrality": [1, 0],
                "bounds": [(0, None), (0, None)],
            },
        )
        self.assertEqual([1.0, 0.5], self._points(sol))
        self.assertAlmostEqual(1.5, self._objective(sol))

    def test_unbounded_variable_with_finite_upper_bound(self):
        sol = self._solve([-1], [], [], parameters={"bounds": [("None", 2)]})
        self.assertEqual([2.0], self._points(sol))
        self.assertAlmostEqual(-2.0, self._objective(sol))

    def test_solution_vector_order(self):
        expected = list(range(12))
        sol = self._solve([1] * 12, [], [], np.eye(12).tolist(), expected)
        self.assertEqual(expected, self._points(sol))
        self.assertAlmostEqual(sum(expected), self._objective(sol))

    def test_infeasible_problem_and_fallback_values(self):
        sol = self._solve([1], [[1], [-1]], [0, -1])
        self.assertIsNone(self._points(sol))
        for maximize, expected in ((False, sys.float_info.min), (True, sys.float_info.max)):
            with self.subTest(maximize=maximize):
                self.assertEqual(
                    [expected] * 2,
                    self._points(sol, {"return_when_none": True, "maximize": maximize,
                                       "var_corr": {"x": 0, "y": 1}}),
                )

    def test_variable_parameter_lengths(self):
        if solver.PULP not in solver.VERSIONS_APPLY:
            self.skipTest("PuLP is not installed")
        for parameter, value in (("integrality", [1]), ("bounds", [(0, None)])):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    self._solve([1, 1], [], [], parameters={parameter: value})

    def test_v4_numpy_inputs_and_column_rhs(self):
        self._v4_backend()
        for array_type in (np.array, np.matrix):
            with self.subTest(array_type=array_type):
                sol = self._solve(
                    np.array([1, 2], dtype=np.int64),
                    array_type([[-1, -1]], dtype=np.int64),
                    array_type([[-2]], dtype=np.int64),
                    array_type([[0, 1]], dtype=np.int64),
                    array_type([[1]], dtype=np.int64),
                    parameters={"bounds": np.array([[0, 3], [0, 3]], dtype=np.int64)},
                )
                self.assertEqual([1.0, 1.0], self._points(sol))
                self.assertAlmostEqual(3.0, self._objective(sol))

    def test_v4_zero_objective_excludes_dummy_variable(self):
        self._v4_backend()
        sol = self._solve([0, 0], [], [], [[1, 0], [0, 1]], [2, 3])
        self.assertEqual([2.0, 3.0], self._points(sol))
        self.assertEqual(0.0, self._objective(sol))
        self.assertTrue(sol.success)

    def test_v4_nonoptimal_solution_is_rejected(self):
        backend = self._v4_backend()
        sol = self._solve([1], [[-1]], [-1])
        sol.stats = replace(sol.stats, status=backend.LpSolveStatus.TimeLimit, has_solution=True)
        self.assertFalse(sol.success)
        self.assertIsNone(self._points(sol))

    def test_v4_missing_cbc_message(self):
        backend = self._v4_backend()
        with mock.patch.object(backend.COIN_CMD, "available", return_value=False):
            with self.assertRaisesRegex(backend.PulpSolverError, r"pulp\[cbc\]"):
                self._solve([1], [[-1]], [-1])

    def test_approximated_process_tree_alignments(self):
        if solver.PULP not in solver.VERSIONS_APPLY:
            self.skipTest("PuLP is not installed")
        import pm4py
        from pm4py.algo.conformance.alignments.process_tree import algorithm
        from pm4py.objects.log.obj import Event, EventLog, Trace

        scenarios = (
            ("->( 'A', 'B', 'C' )", ("A", "B", "C")),
            ("+( 'A', 'B' )", ("A", "B")),
            ("*( 'A', 'B' )", ("A", "B", "A")),
        )
        for tree_string, activities in scenarios:
            with self.subTest(tree=tree_string):
                trace = Trace([Event({"concept:name": activity}) for activity in activities])
                result = algorithm.apply(
                    EventLog([trace]), pm4py.parse_process_tree(tree_string),
                    variant=algorithm.Variants.APPROXIMATED_ORIGINAL,
                    parameters={"max_trace_length": 0, "max_process_tree_height": 1},
                )
                self.assertEqual(1, len(result))
                self.assertEqual(0, result[0]["cost"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import multiprocessing
import sys
import unittest
from unittest import mock

HAS_CVXOPT = importlib.util.find_spec("cvxopt") is not None

# Two small integer problems that abort GLPK 4.65 (bundled in the cvxopt Linux
# wheel) once integer presolve removes every column. Bounds 0..4 are passed as
# inequality rows, as pm4py's process tree MILP does for this variant.
CASES = {
    "seeded_integer_03": ([5, 1, 4], [[0, 3, -2], [2, 3, 0], [2, -2, 0], [0, 1, -1], [2, 1, -1]], [1, 9.25, -6, 0, -1], 19.0),
    "seeded_integer_15": ([1, 4, 5], [[1, -2, 1], [3, 3, -2], [1, 0, 3], [-1, 2, 0], [-2, 0, 2]], [-3, 3.25, 9, 7, 6], 27.0),
}

def _solve_case(connection, c, source_rows, source_rhs):
    from cvxopt import matrix
    from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp

    rows, rhs = list(source_rows), list(source_rhs)
    for i in range(len(c)):
        rows += [[-1.0 if j == i else 0.0 for j in range(len(c))],
                 [1.0 if j == i else 0.0 for j in range(len(c))]]
        rhs += [0.0, 4.0]
    G = matrix([[float(r[j]) for r in rows] for j in range(len(c))])
    sol = ilp.custom_solve_ilp(matrix([float(v) for v in c]), G,
                             matrix([float(v) for v in rhs]), matrix(0.0, (0, len(c))),
                             matrix(0.0, (0, 1)), set(range(len(c))))
    connection.send({"status": sol["status"],
                     "x": list(sol["x"]) if sol["x"] is not None else None,
                     "obj": sol["primal objective"]})
    connection.close()


@unittest.skipUnless(HAS_CVXOPT, "cvxopt is not installed")
class CvxoptGlpkGuardTest(unittest.TestCase):
    def test_zero_column_presolve_cases_do_not_abort(self):
        context = multiprocessing.get_context("spawn")
        for name, (c, rows, rhs, expected) in CASES.items():
            with self.subTest(case=name):
                # A child process, because the unguarded failure is SIGABRT.
                receive, send = context.Pipe(duplex=False)
                proc = context.Process(target=_solve_case, args=(send, c, rows, rhs))
                try:
                    proc.start()
                    send.close()
                    proc.join(timeout=120)
                    self.assertFalse(proc.is_alive(), "integer solve timed out")
                    self.assertEqual(proc.exitcode, 0, "integer solve crashed")
                    self.assertTrue(receive.poll(), "integer solve returned no result")
                    out = receive.recv()
                finally:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                    receive.close()
                    send.close()
                self.assertEqual(out["status"], "optimal")
                x = out["x"]
                self.assertTrue(all(abs(v - round(v)) <= 1e-9 and 0 <= v <= 4 for v in x))
                for row, bound in zip(rows, rhs):
                    self.assertLessEqual(sum(a * v for a, v in zip(row, x)), bound + 1e-9)
                self.assertAlmostEqual(out["obj"], sum(a * v for a, v in zip(c, x)))
                self.assertAlmostEqual(out["obj"], expected)

    def _route(self, version, relaxation=(0.5, 1.0)):
        from cvxopt import matrix
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        fake_glpk = mock.Mock()
        fake_glpk.lp.return_value = ("optimal", matrix(list(relaxation)), None, None)
        fake_glpk.ilp.return_value = ("optimal", matrix([1.0, 1.0]))
        with mock.patch.object(ilp, "glpk", fake_glpk), \
                mock.patch.object(ilp, "_loaded_glpk_version", return_value=version) as detect, \
                mock.patch.object(ilp, "_solve_ilp_with_scipy", return_value=("optimal", matrix([1.0, 1.0]))) as fallback:
            sol = ilp.custom_solve_ilp(matrix([1.0, 1.0]), matrix([[-1.0], [-1.0]]), matrix([-1.5]),
                                       matrix(0.0, (0, 2)), matrix(0.0, (0, 1)), {0, 1})
        return sol, fake_glpk, detect, fallback

    def test_glpk_older_than_5_uses_the_scipy_fallback(self):
        sol, fake_glpk, _, fallback = self._route((4, 65))
        fallback.assert_called_once()
        fake_glpk.ilp.assert_not_called()
        self.assertEqual(sol["status"], "optimal")
        self.assertAlmostEqual(sol["primal objective"], 2.0)

    def test_glpk_5_and_unknown_versions_keep_glpk(self):
        for version in ((5, 0), (5, 1), None):
            with self.subTest(version=version):
                _, fake_glpk, _, fallback = self._route(version)
                fake_glpk.ilp.assert_called_once()
                fallback.assert_not_called()

    def test_integral_relaxation_skips_detection_and_integer_solvers(self):
        sol, fake_glpk, detect, fallback = self._route((4, 65), relaxation=(1.0, 1.0))
        detect.assert_not_called()
        fallback.assert_not_called()
        fake_glpk.ilp.assert_not_called()
        self.assertEqual(list(sol["x"]), [1.0, 1.0])

    def test_fallback_keeps_variables_free_and_rounds_only_integers(self):
        from cvxopt import matrix, spmatrix
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        # Minimise x0 - x1 with integer x0 in [-4.5, -2.5], continuous x1 in [0, 0.25] and x0 + 4 x1 = -3.
        # The optimum x0 = -4, x1 = 0.25 needs a negative variable and an unrounded continuous one.
        c = matrix([1.0, -1.0])
        G = spmatrix([1.0, -1.0, 1.0, -1.0], [0, 1, 2, 3], [0, 0, 1, 1])
        h = matrix([-2.5, 4.5, 0.25, 0.0])
        status, x = ilp._solve_ilp_with_scipy(c, G, h, matrix([[1.0], [4.0]]), matrix([-3.0]), {0})
        self.assertEqual(status, "optimal")
        self.assertEqual(x.size, (2, 1))
        self.assertEqual(x[0], -4.0)
        self.assertAlmostEqual(x[1], 0.25)

    def test_fallback_rejects_index_sets_that_glpk_rejects(self):
        from cvxopt import matrix
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        for I, error in (({-1}, IndexError), ({2}, IndexError), ([0], TypeError), ({0.5}, TypeError)):
            with self.subTest(I=I):
                with self.assertRaises(error):
                    ilp._solve_ilp_with_scipy(matrix([1.0, 1.0]), matrix([[-1.0], [-1.0]]), matrix([-1.5]), None, None, I)

    def test_fallback_treats_bool_elements_as_indices(self):
        from cvxopt import matrix
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        # Minimise x with 0.5 <= x <= 1: integer index 0 (given as False) gives 1, a continuous x gives 0.5.
        status, x = ilp._solve_ilp_with_scipy(matrix([1.0]), matrix([-1.0, 1.0]), matrix([-0.5, 1.0]), None, None, {False})
        self.assertEqual((status, x[0]), ("optimal", 1.0))
        status, x = ilp._solve_ilp_with_scipy(
            matrix([0.0, 1.0]), matrix([[0.0, 0.0], [-1.0, 1.0]]), matrix([-0.5, 1.0]), None, None, {True})
        self.assertEqual((status, x[1]), ("optimal", 1.0))

    def test_detector_takes_the_lowest_visible_version(self):
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        cases = (((4, 65), (5, 0), (4, 65)), ((5, 0), (4, 65), (4, 65)), ((5, 0), None, (5, 0)), (None, None, None))
        for linked, global_scope, expected in cases:
            with self.subTest(linked=linked, global_scope=global_scope):
                ilp._loaded_glpk_version.cache_clear()
                lookups = {"linked": linked, "global": global_scope}
                with mock.patch.object(ilp.sys, "platform", "linux"), \
                        mock.patch.object(ilp, "_glpk_version_of", side_effect=lambda path: lookups["global" if path is None else "linked"]):
                    self.assertEqual(ilp._loaded_glpk_version(), expected)
        ilp._loaded_glpk_version.cache_clear()

    @unittest.skipUnless(sys.platform.startswith("linux"), "the GLPK version is only detected on Linux")
    def test_detector_returns_a_version_or_none_on_linux(self):
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        version = ilp._loaded_glpk_version()
        if version is not None:
            self.assertEqual(len(version), 2)
            self.assertTrue(all(isinstance(v, int) for v in version))

    def test_fallback_reports_integer_infeasibility(self):
        from cvxopt import matrix
        from pm4py.util.lp.variants import cvxopt_solver_custom_align_ilp as ilp
        status, x = ilp._solve_ilp_with_scipy(
            matrix([1.0]), matrix([-1.0, 1.0]), matrix([-0.25, 0.75]), None, None, {0})
        self.assertEqual((status, x), ("infeasible problem", None))


if __name__ == "__main__":
    unittest.main()

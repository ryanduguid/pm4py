'''
PM4Py – A Process Mining Library for Python
Copyright (C) 2026 Process Intelligence Solutions GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
'''
import ctypes
import os
import re
import sys
from enum import Enum
from functools import lru_cache
from threading import Lock

import numpy as np

from pm4py.util import exec_utils

from cvxopt import blas
from cvxopt import glpk
from cvxopt import matrix


class Parameters(Enum):
    INTEGRALITY = "integrality"


this_options = {}
this_options["LPX_K_MSGLEV"] = 0
this_options["msg_lev"] = "GLP_MSG_OFF"
this_options["show_progress"] = False
this_options["presolve"] = "GLP_ON"

this_options_lp = {}
this_options_lp["LPX_K_MSGLEV"] = 0
this_options_lp["msg_lev"] = "GLP_MSG_OFF"
this_options_lp["show_progress"] = False
this_options_lp["presolve"] = "GLP_ON"

TOL = 10**(-5)

LP_LOCK = Lock()


def check_lp_sol_is_integer(x):
    for i in range(len(x)):
        if abs(x[i] - round(x[i])) > TOL:
            return False
    return True


def _glpk_version_of(path):
    try:
        library = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_LOCAL", 0))
        version_fn = library.glp_version
        version_fn.argtypes = []
        version_fn.restype = ctypes.c_char_p
        match = re.match(rb"\s*(\d+)\.(\d+)", version_fn() or b"")
    except (OSError, AttributeError, ValueError):
        return None
    return (int(match.group(1)), int(match.group(2))) if match else None


@lru_cache(maxsize=1)
def _loaded_glpk_version():
    """
    Returns the lowest (major, minor) GLPK version visible to cvxopt.glpk, or
    None when none can be determined (only Linux is checked). It looks up
    glp_version through the extension's own handle, which finds its linked
    GLPK even when another one is installed, and through the process-global
    scope, whose GLPK takes precedence for cvxopt's calls if one was loaded
    globally first. Taking the lowest is conservative when the two differ.
    """
    if not sys.platform.startswith("linux"):
        return None
    module_path = getattr(glpk, "__file__", None)
    versions = [v for v in (_glpk_version_of(module_path) if module_path else None, _glpk_version_of(None)) if v]
    return min(versions) if versions else None


def _to_scipy(mat):
    from scipy.sparse import coo_matrix
    if isinstance(mat, matrix):
        return np.array(mat, dtype=float).reshape(mat.size)
    return coo_matrix((list(mat.V), (list(mat.I), list(mat.J))), shape=mat.size).tocsr()


def _solve_ilp_with_scipy(c, G, h, A, b, I):
    """
    Solves the integer problem with SciPy's HiGHS MILP solver, returning the
    same (status, x) shape as cvxopt.glpk.ilp. Variables are free, as in
    cvxopt's GLPK wrapper, and the MIP gap is zero.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise RuntimeError(
            "The GLPK bundled with cvxopt is older than 5.0 and can abort the process on this problem; "
            "install SciPy 1.9 or later, or select the SciPy LP solver variant."
        ) from exc

    n = c.size[0]
    # Same index-set checks and errors as cvxopt.glpk.ilp, so the fallback cannot solve a different model.
    if not isinstance(I, (set, frozenset)):
        raise TypeError("invalid integer index set")
    if not all(isinstance(i, int) for i in I):
        raise TypeError("non-integer element in I")
    if not all(0 <= i < n for i in I):
        raise IndexError("index element out of range in I")
    integer_indices = {int(i) for i in I}  # bool elements are indices to cvxopt, but masks to NumPy
    integrality = np.zeros(n)
    integrality[list(integer_indices)] = 1
    constraints = [LinearConstraint(_to_scipy(G), -np.inf, np.array(h, dtype=float).ravel())]
    if A is not None and A.size[0] > 0:
        beq = np.array(b, dtype=float).ravel()
        constraints.append(LinearConstraint(_to_scipy(A), beq, beq))
    res = milp(
        np.array(c, dtype=float).ravel(), constraints=constraints, integrality=integrality,
        bounds=Bounds(np.full(n, -np.inf), np.full(n, np.inf)),
        options={"disp": False, "presolve": True, "mip_rel_gap": 0.0},
    )
    if res.status == 2:
        return "infeasible problem", None
    if res.status not in (0, 1) or res.x is None:
        return "unknown", None
    x = [float(round(v)) + 0.0 if i in integer_indices and abs(v - round(v)) <= TOL else float(v) for i, v in enumerate(res.x)]
    return ("optimal" if res.status == 0 else "feasible"), matrix(x)


def custom_solve_ilp(c, G, h, A, b, I):
    with LP_LOCK:
        status, x, y, z = glpk.lp(c, G, h, A, b, options=this_options_lp)
        if status == "optimal":
            if not check_lp_sol_is_integer(x):
                version = _loaded_glpk_version()
                if version is not None and version < (5, 0):
                    # GLPK before 5.0 aborts the whole process when integer presolve removes every
                    # column but leaves a row (glp_add_cols with ncs = 0 in ios_create_pool).
                    status, x = _solve_ilp_with_scipy(c, G, h, A, b, I)
                else:
                    status, x = glpk.ilp(c, G, h, A, b, I=I, options=this_options)
            if status == 'optimal':
                pcost = blas.dot(c, x)
            else:
                pcost = None

            return {'status': status, 'x': x, 'primal objective': pcost}
        else:
            return {'status': status, 'x': None, 'primal objective': None}


def apply(c, Aub, bub, Aeq, beq, parameters=None):
    """
    Gets the overall solution of the problem

    Parameters
    ------------
    c
        c parameter of the algorithm
    Aub
        A_ub parameter of the algorithm
    bub
        b_ub parameter of the algorithm
    Aeq
        A_eq parameter of the algorithm
    beq
        b_eq parameter of the algorithm
    parameters
        Possible parameters of the algorithm

    Returns
    -------------
    sol
        Solution of the LP problem by the given algorithm
    """
    if parameters is None:
        parameters = {}

    integrality = exec_utils.get_param_value(Parameters.INTEGRALITY, parameters, None)

    if integrality is None:
        size = Aub.size[1]
        I = {i for i in range(size)}
    else:
        I = {i for i in range(len(integrality)) if integrality[i] == 1}

    sol = custom_solve_ilp(c, Aub, bub, Aeq, beq, I)

    return sol


def get_prim_obj_from_sol(sol, parameters=None):
    """
    Gets the primal objective from the solution of the LP problem

    Parameters
    -------------
    sol
        Solution of the ILP problem by the given algorithm
    parameters
        Possible parameters of the algorithm

    Returns
    -------------
    prim_obj
        Primal objective
    """
    return sol["primal objective"]


def get_points_from_sol(sol, parameters=None):
    """
    Gets the points from the solution

    Parameters
    -------------
    sol
        Solution of the LP problem by the given algorithm
    parameters
        Possible parameters of the algorithm

    Returns
    -------------
    points
        Point of the solution
    """
    if parameters is None:
        parameters = {}

    maximize = parameters["maximize"] if "maximize" in parameters else False
    return_when_none = parameters["return_when_none"] if "return_when_none" in parameters else False
    var_corr = parameters["var_corr"] if "var_corr" in parameters else {}

    if sol and 'x' in sol and sol['x'] is not None:
        return list(sol['x'])
    else:
        if return_when_none:
            if maximize:
                return [sys.float_info.max] * len(list(var_corr.keys()))
            return [sys.float_info.min] * len(list(var_corr.keys()))

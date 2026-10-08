"""PuLP 4 backend. CBC must be installed, e.g. with ``pip install pulp[cbc]``."""

import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pulp import (
    COIN_CMD,
    LpMinimize,
    LpProblem,
    LpSolveStats,
    LpSolveStatus,
    LpVariable,
    PulpSolverError,
    lpSum,
    value,
)


class Parameters(Enum):
    REQUIRE_ILP = "require_ilp"
    INTEGRALITY = "integrality"
    BOUNDS = "bounds"


MIN_THRESHOLD = 1e-12
MAX_NUM_CONSTRAINTS = 7


@dataclass
class Solution:
    """Keep the model, solve statistics and decision variables together."""

    problem: LpProblem
    stats: LpSolveStats
    variables: list[LpVariable]

    @property
    def success(self):
        return (
            self.stats.status == LpSolveStatus.Optimal
            and self.stats.has_solution
        )


def solver(prob):
    # PM4Py's heuristics need an optimum, rather than a gap-limited incumbent.
    cbc = COIN_CMD(msg=False, gapRel=0, gapAbs=0)
    if not cbc.available():
        raise PulpSolverError(
            "CBC is required by the PM4Py PuLP 4 backend. "
            "Install it with pip install 'pulp[cbc]' or put cbc on PATH."
        )
    return prob.solve(cbc)


def get_variable_name(index):
    return str(index).zfill(MAX_NUM_CONSTRAINTS)


def apply(c, Aub, bub, Aeq=None, beq=None, parameters=None):
    """Solve an LP or ILP and retain the PuLP 4 solve result."""
    # The Rust model requires native scalar values; RHS matrices may be columns.
    c = np.asarray(c).reshape(-1).tolist()
    Aub = np.asarray(Aub).tolist()
    bub = np.asarray(bub).reshape(-1).tolist()
    if Aeq is not None:
        Aeq = np.asarray(Aeq).tolist()
    if beq is not None:
        beq = np.asarray(beq).reshape(-1).tolist()

    if parameters is None:
        parameters = {}

    require_ilp = parameters.get("require_ilp", False)
    integrality = parameters.get("integrality", None)
    bounds = parameters.get("bounds", None)
    num_vars = len(c)

    if integrality is not None and len(integrality) != num_vars:
        raise ValueError(
            "Length of 'integrality' list must be equal to the number of variables."
        )
    if bounds is not None and len(bounds) != num_vars:
        raise ValueError(
            "Length of 'bounds' list must be equal to the number of variables."
        )

    prob = LpProblem("LP_Problem", LpMinimize)
    x_vars = []
    for i in range(num_vars):
        lb, ub = bounds[i] if bounds is not None else (None, None)
        lb = None if lb is None or lb == "None" else float(lb)
        ub = None if ub is None or ub == "None" else float(ub)

        if integrality is not None:
            cat = "Integer" if integrality[i] else "Continuous"
        else:
            cat = "Integer" if require_ilp else "Continuous"

        x_vars.append(
            prob.add_variable(
                f"x_{get_variable_name(i)}", lowBound=lb, upBound=ub, cat=cat
            )
        )

    prob += lpSum(
        c[j] * x_vars[j] for j in range(num_vars) if abs(c[j]) >= MIN_THRESHOLD
    ), "Objective"

    for i in range(len(Aub)):
        constraint_expr = lpSum(
            Aub[i][j] * x_vars[j]
            for j in range(num_vars)
            if abs(Aub[i][j]) >= MIN_THRESHOLD
        )
        prob += constraint_expr <= bub[i], (
            f"Inequality_Constraint_{get_variable_name(i)}"
        )

    if Aeq is not None and beq is not None:
        for i in range(len(Aeq)):
            constraint_expr = lpSum(
                Aeq[i][j] * x_vars[j]
                for j in range(num_vars)
                if abs(Aeq[i][j]) >= MIN_THRESHOLD
            )
            prob += constraint_expr == beq[i], (
                f"Equality_Constraint_{get_variable_name(i + len(Aub))}"
            )

    return Solution(prob, solver(prob), x_vars)


def get_prim_obj_from_sol(sol, parameters=None):
    """Retrieve the objective value, as in the PuLP 3 backend."""
    objective = value(sol.problem.objective)
    if objective is not None:
        return objective
    if sol.success:
        # CBC uses an unset dummy variable for an empty objective expression.
        return 0.0
    return None


def get_points_from_sol(sol, parameters=None):
    """Return decision values in input order, excluding CBC dummy variables."""
    if parameters is None:
        parameters = {}

    if sol.success:
        return [variable.varValue for variable in sol.variables]
    if parameters.get("return_when_none", False):
        default_value = (
            sys.float_info.max
            if parameters.get("maximize", False)
            else sys.float_info.min
        )
        return [default_value] * len(parameters.get("var_corr", {}))
    return None

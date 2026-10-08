import importlib.util

if importlib.util.find_spec("pulp"):
    import pulp

    # PuLP 4 replaces the status constants and the bundled CBC interface.
    if hasattr(pulp, "LpSolveStatus"):
        from pm4py.util.lp.variants import pulp_solver_v4 as pulp_solver
    else:
        from pm4py.util.lp.variants import pulp_solver

if importlib.util.find_spec("scipy"):
    from pm4py.util.lp.variants import scipy_solver

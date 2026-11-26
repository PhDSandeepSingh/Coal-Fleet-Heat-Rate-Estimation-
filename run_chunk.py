import nbformat, os
nb = nbformat.read("Paper2_CoalHeatRate.ipynb", as_version=4); ns = {"__name__": "__main__", "display": print}
for c in nb.cells:
    if c.cell_type != "code": continue
    if "Parallel(n_jobs" in c.source: exec(c.source, ns); break
    exec(c.source, ns)
print("done files:", len(list((ns["OUT"] / "runs").glob("*.pkl"))))

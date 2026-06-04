"""Validate the engine against documented SimCity-classic scenario populations.

Loads the 8 classic disaster-scenario save files (downloaded from the
MicropolisCore repo, vendored under engine/cities/scenario_*.cty) and checks
that the engine's recomputed cityPop matches the populations documented in
the SimCity community wikis. This is an engine-faithfulness check: a saved
map should reproduce its known population.

    /usr/bin/python3 validate_scenarios.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slimcity import MicropolisEnv

# Documented starting populations (SimCity classic, PC). '-' = not found.
DOCUMENTED = {
    "rio_de_janeiro": 152480,
    "bern":            95200,
    "boston":          77520,
    "detroit":         76040,
    "tokyo":           None,
    "hamburg":         None,
    "san_francisco":   None,
    "dullsville":      None,
}

def pop_of(path):
    env = MicropolisEnv(seed=42, load_city=path)
    e = env.engine
    e.setSpeed(3); e.setPasses(1); e.setEnableDisasters(False)
    e.simTick()                      # run the census once
    return int(e.cityPop)

def main():
    cdir = os.path.join(HERE, "engine", "cities")
    print(f"{'scenario':18s} {'engine':>9s} {'documented':>11s} {'delta':>8s}")
    for name, doc in DOCUMENTED.items():
        p = os.path.join(cdir, f"scenario_{name}.cty")
        if not os.path.exists(p):
            print(f"{name:18s} {'(missing)':>9s}")
            continue
        got = pop_of(p)
        if doc is None:
            print(f"{name:18s} {got:>9d} {'—':>11s} {'—':>8s}")
        else:
            d = got - doc
            pct = 100.0 * d / doc
            print(f"{name:18s} {got:>9d} {doc:>11d} {d:>+6d} ({pct:+.1f}%)")

if __name__ == "__main__":
    main()

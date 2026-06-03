"""ELM (Evolution through Large Models) harness for slimcity.

A MAP-Elites loop whose genome is *Python source code* (a closed-loop
placement policy) and whose variation operator is an LLM (Claude). No float
vectors / CMA-ES here — see the top-level qd_train.py for that path.

Modules:
    sandbox        — compile + safely run an evolved policy; the primitives
                     the evolved code (and the LLM) are given.
    evaluate_code  — roll a code genome out on MicropolisEnv -> fitness/measures.
    archive        — dict-based MAP-Elites over code genomes.
    seeds          — hand-written starter genomes.
    operator       — LLM mutation/crossover (+ a mock for offline testing).
    elm_train      — the driver loop / CLI.
"""

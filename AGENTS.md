# Agent notes (slimcity)

Project root for all commands. Read **EXPERIMENTS.md** before starting training; read **RESULTS.md** for what already works.

**Smoke:** `python3 random_smoke.py` then `python3 qd_train.py --gens 5 --emitters 1 --batch 8`

**Long runs:** use `--save results/<unique_name>.npz` and log the run in EXPERIMENTS.md.

**Visualize:** `python3 record.py --mode policy --archive <path>.npz --frames 200 --out results/<name>.gif`

Parallel experiments on other machines should use distinct `results/` filenames.

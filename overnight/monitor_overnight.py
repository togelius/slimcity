#!/usr/bin/env python3
"""Overnight watchdog for the ELM run: every INTERVAL minutes it parses the run
log, emails julian@togelius.com a progress summary, and commits+pushes the
latest results to main so they can be pulled from another machine.

Usage (detached):
    PYTHONPATH=. nohup /usr/bin/python3 overnight/monitor_overnight.py \
        --log results/elm_comb1000_overnight.log \
        --npz results/elm_comb1000_overnight.npz \
        --pid-file overnight/elm.pid --interval 45 &
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time, tempfile

TO = "julian@togelius.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, timeout=120):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def pid_alive(pid_file):
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def parse_log(log_path):
    """Pull the latest progress numbers out of the run log."""
    s = dict(iter=0, best_pop=0, best_fit=0.0, filled=0, evals=0, op_errors=0,
             invalid=0, last="(no eval lines yet)")
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return s
    for ln in lines:
        m = re.search(r"best_pop=(\d+)", ln)
        if m:
            s["best_pop"] = max(s["best_pop"], int(m.group(1)))
        m = re.search(r"best_fit=([\d.]+)", ln)
        if m:
            s["best_fit"] = max(s["best_fit"], float(m.group(1)))
        m = re.search(r"filled=(\d+)", ln)
        if m:
            s["filled"] = int(m.group(1))
        m = re.match(r"it\s+(\d+)\s", ln)
        if m:
            s["iter"] = max(s["iter"], int(m.group(1)))
        if " op-error" in ln:
            s["op_errors"] += 1
        if "invalid:" in ln:
            s["invalid"] += 1
        if re.search(r"\bfit=.*pop=.*cell=", ln):
            s["evals"] += 1
            s["last"] = ln.strip()[:200]
    return s


def send_email(subject, body):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(body)
        path = tf.name
    try:
        r = sh(["/usr/bin/python3", "-m", "elm.send_email", TO, subject, path], timeout=90)
        ok = r.returncode == 0
        if not ok:
            print("email failed:", r.stdout[-300:], r.stderr[-300:], flush=True)
        return ok
    finally:
        os.unlink(path)


def commit_push(npz, log, msg):
    best_py = npz.rsplit(".", 1)[0] + "_best.py"
    for f in (npz, log, best_py):
        if os.path.exists(os.path.join(ROOT, f)):
            sh(["git", "add", f])
    # only commit if something is staged
    if sh(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        sh(["git", "commit", "-m", msg])
    r = sh(["git", "pull", "--rebase", "origin", "main"], timeout=180)
    if r.returncode != 0:                       # don't leave a half-finished rebase
        sh(["git", "rebase", "--abort"])
        sh(["git", "merge", "--abort"])
        return r
    return sh(["git", "push", "origin", "main"], timeout=180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--interval", type=float, default=45.0, help="minutes")
    a = ap.parse_args()

    t0 = time.time()
    prev = dict(best_pop=0, iter=0)
    n = 0
    while True:
        alive = pid_alive(a.pid_file)
        time.sleep(2)
        s = parse_log(a.log)
        hrs = (time.time() - t0) / 3600.0
        dpop = s["best_pop"] - prev["best_pop"]
        ditr = s["iter"] - prev["iter"]
        status = "RUNNING" if alive else "STOPPED"
        subject = f"[slimcity ELM] {status} | best={s['best_pop']} | it={s['iter']} | {hrs:.1f}h"
        body = (
            f"slimcity overnight ELM run ({status})\n"
            f"episode: 1000 actions x 10 ticks, measures=cl_ind, seeded with comb1000 + top closed-loop champions\n\n"
            f"  wall time     : {hrs:.1f} h\n"
            f"  iterations    : {s['iter']}  (+{ditr} since last report)\n"
            f"  BEST cityPop  : {s['best_pop']}  (+{dpop} since last report)\n"
            f"  best fitness  : {s['best_fit']:.0f}\n"
            f"  archive filled: {s['filled']} cells\n"
            f"  evals / op-err / invalid : {s['evals']} / {s['op_errors']} / {s['invalid']}\n\n"
            f"  baseline to beat: prior closed-loop champion ~19,500 @1000x10; comb seed ~83,000\n\n"
            f"  latest: {s['last']}\n\n"
            f"  (results committed to main: results/{os.path.basename(a.npz)} + _best.py; pull to inspect)\n"
        )
        send_email(subject, body)
        cp = commit_push(a.npz, a.log,
                         f"ELM overnight checkpoint: it={s['iter']} best_pop={s['best_pop']} filled={s['filled']} ({hrs:.1f}h)")
        print(f"[{time.strftime('%H:%M')}] report #{n}: {status} best={s['best_pop']} it={s['iter']} "
              f"push_rc={cp.returncode}", flush=True)
        prev = dict(best_pop=s["best_pop"], iter=s["iter"])
        n += 1
        if not alive:
            print("ELM process gone — final report sent, monitor exiting.", flush=True)
            break
        time.sleep(max(60.0, a.interval * 60.0 - 2))


if __name__ == "__main__":
    main()

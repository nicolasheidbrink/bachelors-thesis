'''
This script was simultaneously run on three separate jobs on the HAICORE cluster, 
    with 4 GPUs each.

It finds copula pairs from `unique_pairs.json` that have not been computed, 
    computes the distance, and writes the distance to the file.

It uses locks on the json file to ensure that calculations are not lost. 

This script, particularly the logistics behind file locks etc. starting at (2), was 
    implemented with Claude Code.
'''

import argparse
import fcntl
import getpass
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import ot

from distance_calculation_campaign.copula_families import MirroredCopula, TransposedCopula
from distance_calculation_campaign.unique_pairs import unique_shapes
from distance_calculation_campaign.compute_distance import run_transport

# --------------------------------------------------------------------------
# 1. Calculating distances
# --------------------------------------------------------------------------

QUEUE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_pairs.json")

QUEUED = "queued"
INPROGRESS = "inprogress"
COMPLETED = "completed"
FAILED = "failed"

# Used to apply individual operations to copulas
isometries = {
    "M0": lambda wrappers, copula: wrappers["Mirrored"](copula, 0),
    "M1": lambda wrappers, copula: wrappers["Mirrored"](copula, 1),
    "M01": lambda wrappers, copula: wrappers["Mirrored"](copula, (0, 1)),
    "T": lambda wrappers, copula: wrappers["Transposed"](copula),
}

shapes = unique_shapes
wrappers = {"Mirrored": MirroredCopula, "Transposed": TransposedCopula}

# Parses copula from string encoding
def build_copula(name):
    head, sep, rest = name.partition("!")
    copula = shapes[head]

    for tag in (rest.split("!") if sep else []):
        copula = isometries[tag](wrappers, copula)
    return copula

# Sample-based calculation of Wasserstein distance
def regular_W2(copula0, copula1, *, npts=5000, seed=0):
    np.random.seed(seed)
    u0 = copula0.sample_u(npts)
    u1 = copula1.sample_u(npts)
    M = ot.dist(u0, u1)
    a = np.ones(npts) / npts
    b = np.ones(npts) / npts
    loss, log = ot.emd2(a, b, M, numItermax=1_000_000, log=True)
    return math.sqrt(loss)

def compute_pair(name_a, name_b):
    copula0 = build_copula(name_a)
    copula1 = build_copula(name_b)

     # calculates copula-constrained Wasserstein distance using `compute_distance.py`,
     #    which returns a dict with the distance and other relevant values:
    results = run_transport(copula0, copula1)

    # adds the regular Wasserstein distance to the dict:
    results["regular_W2"] = regular_W2(copula0, copula1)

    return results

# --------------------------------------------------------------------------
# 2. Locking and atomic writes
# --------------------------------------------------------------------------

@contextmanager
def queue_lock(lock_path, *, timeout=600):
    """Exclusive lock over a whole read-modify-write cycle.

    Held on a dedicated file, never on the JSON itself. `flock` is released by
    the kernel when the fd closes, so a worker that gets SIGKILLed while
    holding it leaves no stale lock -- the main reason to prefer it over a
    hand-rolled lockfile here.

    Every holder does a read plus one small rename, so waits are milliseconds
    even with a dozen workers; the timeout exists only so a pathological stall
    fails loudly instead of hanging a GPU forever.
    """
    deadline = time.monotonic() + timeout
    f = open(lock_path, "w")
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"could not acquire {lock_path} within {timeout}s"
                    )
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def read_rows(queue_path):
    with open(queue_path) as f:
        return json.load(f)


def write_rows_atomic(queue_path, rows):
    """Write the queue so that a reader sees either the whole old file or the
    whole new one, never a torn one. Call only while holding the lock."""
    directory = os.path.dirname(os.path.abspath(queue_path))
    tmp = os.path.join(directory, f".{os.path.basename(queue_path)}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())          # get the bytes onto disk, not just page cache
    os.replace(tmp, queue_path)       # atomic rename
    # ...and make the rename itself survive a node crash.
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# --------------------------------------------------------------------------
# 3. Reclaiming rows stranded by a killed job
# --------------------------------------------------------------------------

def live_job_ids():
    """Job ids currently known to Slurm, or None if we could not find out.

    None is load-bearing: if `squeue` is missing or errors, we must NOT
    conclude that every claiming job is dead and requeue rows out from under
    running workers. Callers treat None as "skip the sweep this time".

    An *empty set* is a different answer and a legitimate one -- squeue
    answered and nothing is running, so every inprogress row really is stale.
    """
    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # "%i" renders array tasks as 1234_5; the base id is what we store.
    return {line.strip().split("_")[0] for line in out.stdout.splitlines() if line.strip()}


def requeue_stale(rows, live, *, my_jobid=None):
    """Reset inprogress rows whose claiming job is gone. Returns the pairs reset.

    `live` is the set from `live_job_ids()`, or None for "could not tell", in
    which case nothing is touched. It is a required argument on purpose: when
    it defaulted to None and the function queried squeue itself, None meant
    both "go look it up" and "lookup failed", so the guard below could not be
    reached from outside and a caller passing None to mean "unknown" got a
    full sweep instead.

    Mutates `rows`; the caller writes them back under the lock.
    """
    if live is None:
        return []
    reset = []
    for row in rows:
        if row.get("status") != INPROGRESS:
            continue
        jobid = str(row.get("jobid") or "")
        if not jobid:
            continue                  # claimed without a jobid: not ours to judge
        if jobid == str(my_jobid) or jobid in live:
            continue                  # claiming job still alive
        row["status"] = QUEUED
        row["requeued_from"] = jobid
        row["requeued_at"] = _now()
        for key in ("jobid", "rank", "host", "claimed_at"):
            row.pop(key, None)
        reset.append(row["pair"])
    return reset


# --------------------------------------------------------------------------
# 4. Claim / release / record
# --------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def claim_next(queue_path, lock_path, *, jobid, rank, host, sweep=True):
    """Atomically take the first queued pair. Returns its pair string, or None.

    The stale sweep runs inside the same lock as the claim, so a row freed by
    the sweep is immediately available to the very worker that freed it.
    """
    with queue_lock(lock_path):
        rows = read_rows(queue_path)
        dirty = False

        if sweep:
            reset = requeue_stale(rows, live_job_ids(), my_jobid=jobid)
            if reset:
                dirty = True
                log(f"requeued {len(reset)} stale pair(s) from dead job(s): "
                    + ", ".join(reset[:4]) + (" ..." if len(reset) > 4 else ""))

        claimed = None
        for row in rows:
            if row.get("status") == QUEUED:
                row["status"] = INPROGRESS
                row["jobid"] = jobid
                row["rank"] = rank
                row["host"] = host
                row["claimed_at"] = _now()
                claimed = row["pair"]
                dirty = True
                break

        if dirty:
            write_rows_atomic(queue_path, rows)
        return claimed


def finish(queue_path, lock_path, pair, status, payload):
    """Write a pair's outcome back. `payload` is merged into the row."""
    with queue_lock(lock_path):
        rows = read_rows(queue_path)
        for row in rows:
            if row["pair"] == pair:
                row.update(payload)
                row["status"] = status
                row["finished_at"] = _now()
                break
        else:
            log(f"WARNING: {pair} vanished from the queue; result not stored")
            return
        write_rows_atomic(queue_path, rows)


def release(queue_path, lock_path, pair):
    """Hand a claimed pair back unstarted -- used on shutdown, so the next job
    picks it up without waiting for the stale sweep."""
    with queue_lock(lock_path):
        rows = read_rows(queue_path)
        for row in rows:
            if row["pair"] == pair and row.get("status") == INPROGRESS:
                row["status"] = QUEUED
                for key in ("jobid", "rank", "host", "claimed_at"):
                    row.pop(key, None)
                write_rows_atomic(queue_path, rows)
                break


# --------------------------------------------------------------------------
# 5. Shutting down before the wall limit
# --------------------------------------------------------------------------

class ShutdownRequested(Exception):
    pass


_shutdown = False


def _on_signal(signum, frame):
    """Slurm's wall-limit SIGTERM, or the SIGUSR1 warning shot if the batch
    script asked for one. Raising unwinds out of the training call so the
    claim can be released; the flag stops the loop taking another pair."""
    global _shutdown
    _shutdown = True
    raise ShutdownRequested(f"signal {signum}")


def job_end_time():
    """Epoch seconds at which Slurm will kill this job, or None if unknown."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end:
        try:
            return float(end)
        except ValueError:
            pass
    jobid = os.environ.get("SLURM_JOB_ID")
    if not jobid:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", "-o", jobid],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for field in out.stdout.split():
        if field.startswith("EndTime="):
            try:
                stamp = datetime.strptime(field[len("EndTime="):], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                return None
            return stamp.timestamp()
    return None


# --------------------------------------------------------------------------
# 6. Worker loop
# --------------------------------------------------------------------------

RANK = "?"
HOST = socket.gethostname()


def log(msg):
    print(f"[{_now()} {HOST} rank {RANK}] {msg}", flush=True)


def worker(args):
    global RANK
    jobid = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    RANK = args.rank
    lock_path = args.queue + ".lock"

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    end_time = job_end_time()
    if end_time is not None:
        log(f"job ends at {datetime.fromtimestamp(end_time).isoformat(timespec='seconds')} "
            f"({(end_time - time.time()) / 3600:.1f} h from now)")
    log(f"job {jobid}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    done = 0
    pair = None
    try:
        while True:
            if _shutdown:
                log("shutdown requested; not taking another pair")
                break
            if args.max_pairs and done >= args.max_pairs:
                log(f"reached --max-pairs {args.max_pairs}")
                break
            # Don't start a ~1-2 h run with less than that left on the clock:
            # it would only be killed halfway and requeued.
            if end_time is not None and time.time() + args.min_seconds_left > end_time:
                log(f"under {args.min_seconds_left / 3600:.1f} h left; stopping cleanly")
                break

            pair = claim_next(args.queue, lock_path, jobid=jobid, rank=args.rank,
                              host=HOST, sweep=not args.no_sweep)
            if pair is None:
                log("queue empty; exiting")
                break

            name_a, name_b = pair.split("/", 1)
            log(f"START {pair}")
            started = time.time()
            try:
                metrics = compute_pair(name_a, name_b)
            except ShutdownRequested:
                raise
            except Exception:
                runtime = time.time() - started
                log(f"FAILED {pair} after {runtime / 60:.1f} min\n{traceback.format_exc()}")
                finish(args.queue, lock_path, pair, FAILED, {
                    "runtime_s": runtime,
                    "error": traceback.format_exc(limit=20),
                })
                pair = None
                done += 1
                continue

            runtime = time.time() - started
            payload = dict(metrics or {})
            payload.setdefault("runtime_s", runtime)
            payload["jobid"] = jobid
            payload["host"] = HOST
            finish(args.queue, lock_path, pair, COMPLETED, payload)
            log(f"DONE  {pair} in {runtime / 60:.1f} min")
            pair = None
            done += 1

    except (ShutdownRequested, KeyboardInterrupt) as exc:
        log(f"interrupted ({exc})")
        if pair is not None:
            release(args.queue, lock_path, pair)
            log(f"released {pair} back to queued")
    log(f"exiting after {done} pair(s)")
    return 0


# --------------------------------------------------------------------------
# 7. Standalone queue commands
# --------------------------------------------------------------------------

def show_status(args):
    with queue_lock(args.queue + ".lock"):
        rows = read_rows(args.queue)
    counts = {}
    for row in rows:
        counts[row.get("status", "?")] = counts.get(row.get("status", "?"), 0) + 1
    width = max((len(s) for s in counts), default=1)
    print(f"{args.queue}: {len(rows)} pairs")
    for status in sorted(counts):
        print(f"  {status:<{width}}  {counts[status]}")

    runtimes = [row["runtime_s"] for row in rows
                if row.get("status") == COMPLETED and "runtime_s" in row]
    if runtimes:
        total = sum(runtimes) / 3600
        mean = total / len(runtimes)
        left = counts.get(QUEUED, 0) + counts.get(INPROGRESS, 0)
        print(f"\n  {total:.1f} GPU-h spent, {mean:.2f} GPU-h/pair mean")
        print(f"  {left} left -> ~{left * mean:.0f} GPU-h "
              f"(~{left * mean / 12:.0f} h wall on 12 GPUs)")

    inflight = [r for r in rows if r.get("status") == INPROGRESS]
    if inflight:
        print(f"\n  in flight:")
        for row in inflight:
            print(f"    {row['pair']}  job {row.get('jobid')} "
                  f"rank {row.get('rank')} on {row.get('host')} since {row.get('claimed_at')}")
    failed = [r for r in rows if r.get("status") == FAILED]
    if failed:
        print(f"\n  failed: " + ", ".join(r["pair"] for r in failed[:10])
              + (" ..." if len(failed) > 10 else ""))
    return 0


def do_requeue_stale(args):
    with queue_lock(args.queue + ".lock"):
        rows = read_rows(args.queue)
        live = live_job_ids()
        if live is None:
            print("could not query squeue; refusing to sweep "
                  "(would requeue rows out from under running workers)")
            return 1
        reset = requeue_stale(rows, live)
        if reset:
            write_rows_atomic(args.queue, rows)
    for pair in reset:
        print(f"requeued {pair}")
    print(f"{len(reset)} pair(s) requeued")
    return 0


def do_check_names(args):
    """Resolve every endpoint name in the queue without training anything.

    Worth running once before submitting: a typo in a shape key or a tag is
    otherwise only discovered when a worker reaches that row, marks it failed,
    and moves on -- possibly hours in, and quietly.
    """
    rows = read_rows(args.queue)
    names = sorted({half for row in rows for half in row["pair"].split("/", 1)})
    bad = []
    for name in names:
        try:
            build_copula(name)
        except Exception as exc:
            bad.append((name, f"{type(exc).__name__}: {exc}"))
    print(f"{len(names)} distinct endpoint names across {len(rows)} pairs")
    for name, err in bad:
        print(f"  FAILED {name}: {err}")
    print("all names resolve" if not bad else f"{len(bad)} name(s) failed to resolve")
    return 1 if bad else 0


def do_retry_failed(args):
    with queue_lock(args.queue + ".lock"):
        rows = read_rows(args.queue)
        n = 0
        for row in rows:
            if row.get("status") == FAILED:
                row["status"] = QUEUED
                row.pop("error", None)
                n += 1
        if n:
            write_rows_atomic(args.queue, rows)
    print(f"{n} failed pair(s) requeued")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", default=QUEUE_DEFAULT)
    # No --gpu / --device flags on purpose. Which GPU this worker uses is
    # decided entirely by CUDA_VISIBLE_DEVICES in the *shell* that starts it:
    #
    #     CUDA_VISIBLE_DEVICES=$i python3 -m distance_calculation_campaign.run_pair_queue --rank $i &
    #
    # A flag could not do it. The training module binds `device` to cuda:0 at
    # import time, and importing this file already pulls that in (via
    # compute_distance) -- by the time argparse runs, torch has probed CUDA and
    # cached the device count, so setting the variable here would change
    # nothing. Within each worker cuda:0 is correct; the shell decides which
    # physical card that name points at.
    parser.add_argument("--rank", default=os.environ.get("SLURM_PROCID", "0"),
                        help="worker id, for logging only")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="stop after this many pairs (0 = until the queue empties)")
    parser.add_argument("--min-seconds-left", type=int, default=10800,
                        help="don't start a new pair with less than this left before "
                             "the job's wall limit (default 3 h)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="don't requeue rows stranded by dead jobs when claiming")
    parser.add_argument("--status", action="store_true", help="print queue summary and exit")
    parser.add_argument("--requeue-stale", action="store_true",
                        help="requeue rows claimed by jobs that are no longer running, then exit")
    parser.add_argument("--retry-failed", action="store_true",
                        help="reset every failed row to queued, then exit")
    parser.add_argument("--check-names", action="store_true",
                        help="resolve every endpoint name in the queue, then exit "
                             "(no training; run this before submitting)")
    args = parser.parse_args(argv)

    if args.check_names:
        return do_check_names(args)
    if args.status:
        return show_status(args)
    if args.requeue_stale:
        return do_requeue_stale(args)
    if args.retry_failed:
        return do_retry_failed(args)

    return worker(args)


if __name__ == "__main__":
    sys.exit(main())

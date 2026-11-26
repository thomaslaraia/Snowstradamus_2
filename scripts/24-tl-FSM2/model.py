"""Prepare an upstream FSM2 binary for a named grid experiment."""
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "FSM2_source"
PIN = "a76945e75b36628161f4679428ed72dce090b368"
OPTIONS = {
    "ALBEDO": (1, 2), "CANINT": (1, 2), "CANMOD": (1, 2),
    "CANRAD": (1, 2), "CANUNL": (1, 2), "CONDCT": (0, 1),
    "DENSTY": (0, 1, 2), "EXCHNG": (0, 1), "HYDROL": (0, 1, 2),
    "SGRAIN": (1, 2), "SNFRAC": (1, 2, 3),
}


def settings(path):
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    if set(cfg) - {"physics", "params", "run", "custom"}:
        raise ValueError("Unknown settings section")
    physics = cfg["physics"]
    if set(physics) != set(OPTIONS) or any(
        type(physics[k]) is not int or physics[k] not in allowed for k, allowed in OPTIONS.items()
    ):
        raise ValueError("Use the physics fields and choices listed in template.toml")
    custom = cfg.get("custom", {})
    if set(custom) - {"loading", "unloading"} or any(
        not isinstance(v, str) or not v.strip() or len(v.splitlines()) != 1 for v in custom.values()
    ):
        raise ValueError("Custom loading and unloading must be single-line Fortran expressions")
    run = cfg.get("run", {})
    cycles = run.get("spinup_cycles", 2)
    if set(run) - {"spinup_cycles"} or type(cycles) is not int or cycles < 0:
        raise ValueError("Expected a nonnegative integer spinup_cycles")
    return cfg


def prepare(path):
    cfg = settings(path)
    cache = ROOT / ".build"
    cache.mkdir(exist_ok=True)
    with (cache / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            work = Path(tmp)
            if not SOURCE.exists():
                clone = work / "clone"
                subprocess.run(["git", "clone", "https://github.com/RichardEssery/FSM2.git", clone], check=True)
                subprocess.run(["git", "-C", clone, "checkout", "--detach", PIN], check=True)
                clone.rename(SOURCE)
            def upstream(file):
                return subprocess.check_output(["git", "-C", SOURCE, "show", f"{PIN}:{file}"], text=True)

            declarations = upstream("src/FSM2_PARAMS.F90").split("namelist /params/")[1].split("!")[0]
            allowed = set(re.findall(r"\w+", declarations.lower()))
            params = cfg.get("params", {})
            if len({k.lower() for k in params}) != len(params) or any(
                k.lower() not in allowed or type(v) not in (int, float) or not math.isfinite(v)
                or (k.lower() == "nhyd" and (type(v) is not int or v < 1))
                for k, v in params.items()
            ):
                raise ValueError("Expected numeric upstream /params/ entries. nhyd requires a positive integer.")
            custom = cfg.get("custom", {})
            key = hashlib.sha256(json.dumps([PIN, cfg["physics"], custom,
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()], sort_keys=True).encode()).hexdigest()
            target = cache / key
            if not target.exists():
                with (work / "src.tar").open("wb") as stream:
                    subprocess.run(["git", "-C", SOURCE, "archive", PIN, "src", "compil.sh"], stdout=stream, check=True)
                subprocess.run(["tar", "-xf", "src.tar"], cwd=work, check=True)
                script = work / "compil.sh"
                code = script.read_text()
                for k, v in dict(cfg["physics"], DRIV1D=2, SWPART=1, ZOFFST=1, PROFNC=0).items():
                    code, count = re.subn(rf"^#define {k} \d+", f"#define {k} {v}", code, flags=re.M)
                    if count != 1:
                        raise ValueError(f"Missing upstream option {k}")
                script.write_text(code)
                intercept = work / "src/INTERCEPT.F90"
                code = intercept.read_text()
                for hook, expr in custom.items():
                    macro = {"loading": "CANINT", "unloading": "CANUNL"}[hook]
                    code, count = re.subn(rf"^#if {macro} == 1\n.*?^#endif", lambda _: "    dsvg = " + expr,
                                          code, flags=re.M | re.S)
                    if count != 1:
                        raise ValueError(f"Missing upstream {macro} block")
                intercept.write_text(code)
                subprocess.run(["bash", "-eu", "compil.sh"], cwd=work, check=True)
                built = work / "built"
                built.mkdir()
                for file in ("FSM2", "src/OPTS.h"):
                    shutil.move(work / file, built)
                built.rename(target)
    return cfg, target / "FSM2"

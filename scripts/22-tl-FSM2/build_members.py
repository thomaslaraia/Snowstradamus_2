"""Build CLM5, Noah-MP and HP98 canopy snow schemes inside FSM2."""
import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

PIN = "a76945e75b36628161f4679428ed72dce090b368"

HP98_UNLOADING_DAYS = 10.0  # Hourly adaptation using the FSM2 timescale.


def members(factors):
    schemes = [
        ("clm5", 1, 4, 6.0),
        ("noahmp", 4, 2, 6.6),
        ("hp98_spruce", 3, 3, 5.9),
        ("hp98_pine", 3, 3, 6.6),
    ]
    for scheme, loading, unloading, capacity in schemes:
        for cap, loss in itertools.product(factors, repeat=2):
            params = {"svai": capacity * cap}
            if unloading == 3:
                params["eunl"] = HP98_UNLOADING_DAYS * 86400 / loss
            else:
                params.update(Tunl=1.87e5 / loss, Uunl=1.56e5 / loss)
            name = f"{scheme}_cap{cap:g}_loss{loss:g}"
            yield name, scheme, loading, unloading, params


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected one occurrence of {old!r}")
    path.write_text(text.replace(old, new))


def configure(root, scheme, loading, unloading, params):
    script = root / "compil.sh"
    for key, old, new in [
        ("DRIV1D", 1, 2), ("SWPART", 0, 1), ("ZOFFST", 0, 1),
        ("CANINT", 1, loading), ("CANUNL", 1, unloading),
    ]:
        replace(script, f"#define {key} {old}", f"#define {key} {new}")

    record = json.dumps(params, sort_keys=True)
    replace(script, "/* Process options",
            f"/* {scheme} parameter defaults: {record} */\n/* Process options")

    src = root / "src"
    defaults = src / "FSM2_PARAMS.F90"
    for key, value in params.items():
        match = re.search(rf"^{key} = [^!\n]+", defaults.read_text(), re.M)
        if not match:
            raise ValueError(f"Missing default for {key}")
        replace(defaults, match[0], f"{key} = {float(value)!r}   ")

    intercept = src / "INTERCEPT.F90"
    if scheme == "clm5":
        # Keep native linear loading with kext=0.5 and capacity=6*VAI.
        replace(intercept, "#elif CANUNL == 2", """#elif CANUNL == 4
    dsvg = (max(Tveg(n)-270.,0.)/Tunl + Ua/Uunl)*dt*Sveg(n)
#elif CANUNL == 2""")
        return

    for filename in ("CANOPY.F90", "FSM2_TIMESTEP.F90"):
        replace(src / filename, "CANOPY(Sveg,Tveg,VAI,",
                "CANOPY(Sveg,Tveg,Ta,VAI,")
    canopy = src / "CANOPY.F90"
    replace(canopy, "real, intent(in) :: &",
            "real, intent(in) :: Ta\nreal, intent(in) :: &")
    density = "67.92 + 51.25*exp((Ta-273.15)/2.59)"
    if scheme == "noahmp":
        density = f"min(120., {density})"
    replace(canopy, "Scap(:) = svai*lveg(:)", f"""Scap(:) = svai*lveg(:)*(0.27 + 46./({density}))
  ! If capacity falls below existing storage, interrupt further loading.
  Scap(:) = max(Scap(:), Sveg(:))""")

    replace(intercept, "#elif CANINT == 2", f"""#elif CANINT == {loading}
    dsvg = max(0.,Scap(n)-Sveg(n))*real(1.d0-exp(-dble(Sfg)*dt/Scap(n)))
    dsvg = min(dsvg,Sfg*dt)
#elif CANINT == 2""")

    if scheme == "noahmp":
        # Eq. 3.4.18 uses canopy ice before this step's interception.
        # FSM2's subsequent cap limits unloading to the surviving solid mass.
        replace(intercept, "drip = 0", "real :: initial_load\n\ndrip = 0")
        replace(intercept, "    ! Interception of falling snow",
                "    initial_load = Sveg(n)\n    ! Interception of falling snow")
        replace(intercept, "0.)/Tunl + Ua/Uunl)*dt*Sveg(n)",
                "0.)/Tunl + Ua/Uunl)*dt*initial_load")
    else:
        replace(intercept, "#elif CANUNL == 2", """#elif CANUNL == 3
    dsvg = Sveg(n)*real(1.d0-exp(-dble(dt)/eunl))
#elif CANUNL == 2""")


def build(source, output, factors):
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, scheme, loading, unloading, params in members(factors):
        final = output / name
        record = dict(name=name, scheme=scheme, commit=PIN, CANINT=loading,
                      CANUNL=unloading, parameters=params, builder_sha256=digest,
                      shared_physics="FSM2 energy balance, sublimation, melt and ground snow")
        if (final / "FSM2").is_file():
            if json.loads((final / "member.json").read_text()) != record:
                raise ValueError(f"{final}: different build; choose a new output directory")
            continue
        print(f"Building {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            root = Path(tmp) / "source"
            subprocess.run(["git", "clone", "-q", "--no-checkout", "--no-hardlinks",
                            str(source.resolve()), str(root)], check=True)
            subprocess.run(["git", "checkout", "-q", "--detach", PIN], cwd=root, check=True)
            configure(root, scheme, loading, unloading, params)
            subprocess.run(["bash", "-eu", "compil.sh"], cwd=root, check=True)
            (root / "member.json").write_text(json.dumps(record, indent=2) + "\n")
            root.rename(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factors", nargs="+", type=float, default=[1.0])
    args = parser.parse_args()
    if any(not math.isfinite(f) or f <= 0 for f in args.factors):
        parser.error("Factors must be finite and positive")
    build(args.source, args.output, sorted(set(args.factors)))

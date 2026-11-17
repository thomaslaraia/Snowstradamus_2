"""Build the S01 pilot executables from clean copies of the pinned source."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

PIN = "a76945e75b36628161f4679428ed72dce090b368"

# name, loading option, unloading option, parameter overrides (FSM2 units)
MEMBERS = [
    (f"fsm2_i{i}_u{u}_sv{s:g}", i, u, {"svai": s})
    for i in (1, 2) for u in (1, 2) for s in (4.4, 5.5)
] + [
    (f"hp98_{tree}_tau{days}d", 3, 3, {"svai": s, "eunl": days * 86400})
    for tree, s in (("pine", 6.6), ("spruce", 5.9)) for days in (5, 10)
]


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected one occurrence of {old!r}")
    path.write_text(text.replace(old, new))


def configure(root, loading, unloading, params):
    script = root / "compil.sh"
    for name, old, new in [
        ("DRIV1D", 1, 2), ("SWPART", 0, 1), ("ZOFFST", 0, 1),
        ("CANINT", 1, loading), ("CANUNL", 1, unloading),
    ]:
        replace(script, f"#define {name} {old}", f"#define {name} {new}")

    # The runner copies OPTS.h into the NetCDF model_options attribute.
    record = json.dumps(params, sort_keys=True)
    replace(script, "/* Process options", f"/* Ensemble parameter defaults: {record} */\n/* Process options")

    src = root / "src"
    path = src / "FSM2_PARAMS.F90"
    for key, value in params.items():
        match = re.search(rf"^{key} = [^!\n]+", path.read_text(), re.M)
        if not match:
            raise ValueError(f"Missing default for {key}")
        replace(path, match[0], f"{key} = {float(value)!r}   ")

    if loading != 3:
        return

    # Supply air temperature to the canopy capacity calculation.
    for name in ("FSM2_TIMESTEP.F90", "CANOPY.F90"):
        replace(src / name, "CANOPY(Sveg,Tveg,VAI,", "CANOPY(Sveg,Tveg,Ta,VAI,")
    canopy = src / "CANOPY.F90"
    replace(canopy, "real, intent(in) :: &", "real, intent(in) :: Ta\nreal, intent(in) :: &")
    replace(canopy, "Scap(:) = svai*lveg(:)", """! HP98 eqs. 11-13. VAI approximates LAI.
  Scap(:) = svai*lveg(:)*(0.27 + 46./(67.92 + 51.25*exp((Ta-273.15)/2.59)))
  ! Capacity limits new loading. Warming alone cannot expel existing snow.
  Scap(:) = max(Scap(:), Sveg(:))""")

    intercept = src / "INTERCEPT.F90"
    replace(intercept, "#elif CANINT == 2", """#elif CANINT == 3
    ! HP98 loading, dense-stand approximation Cp = 1.
    dsvg = max(0.,Scap(n)-Sveg(n)) * real(1.d0-exp(-dble(Sfg)*dt/Scap(n)))
    dsvg = min(dsvg, Sfg*dt)
#elif CANINT == 2""")
    replace(intercept, "#elif CANUNL == 2", """#elif CANUNL == 3
    ! Integrate dL/dt = -L/eunl over one model step.
    dsvg = Sveg(n)*real(1.d0-exp(-dble(dt)/eunl))
#elif CANUNL == 2""")


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, loading, unloading, params in MEMBERS:
        final = output / name
        if (final / "FSM2").is_file():
            continue
        print(f"Building {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            root = Path(tmp) / "source"
            subprocess.run(["git", "clone", "-q", "--no-checkout", "--no-hardlinks",
                            str(source.resolve()), str(root)], check=True)
            subprocess.run(["git", "checkout", "-q", "--detach", PIN], cwd=root, check=True)
            configure(root, loading, unloading, params)
            subprocess.run(["bash", "-eu", "compil.sh"], cwd=root, check=True)
            (root / "member.json").write_text(json.dumps(dict(
                name=name, commit=PIN, CANINT=loading, CANUNL=unloading,
                parameters=params), indent=2) + "\n")
            root.rename(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)

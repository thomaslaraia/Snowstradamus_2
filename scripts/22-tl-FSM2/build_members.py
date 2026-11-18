"""Build canopy-scheme ensemble executables from clean copies of the pinned source."""
import argparse
import json
import hashlib
import itertools
import re
import subprocess
import tempfile
from pathlib import Path

PIN = "a76945e75b36628161f4679428ed72dce090b368"

# Each variant uses FSM2 energy balance, sublimation and melt. VAI is used as L+S.
def members(factors):
    result = []
    for scheme, loading, unloading, capacity in [
        ("clm5", 1, 4, 6.0), ("noahmp", 4, 2, 5.87), ("hp98", 3, 3, 5.87),
    ]:
        for cap, loss in itertools.product(factors, repeat=2):
            params = {"svai": capacity * cap}
            if scheme == "hp98":
                params["eunl"] = 10 * 86400 / loss
            else:
                params.update(Tunl=1.87e5 / loss, Uunl=1.56e5 / loss)
            result.append((f"{scheme}_cap{cap:g}_loss{loss:g}", loading, unloading, params))
    return result


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
    replace(script, "/* Process options",
            f"/* Ensemble parameter defaults: {record} */\n/* Process options")

    src = root / "src"
    path = src / "FSM2_PARAMS.F90"
    for key, value in params.items():
        match = re.search(rf"^{key} = [^!\n]+", path.read_text(), re.M)
        if not match:
            raise ValueError(f"Missing default for {key}")
        replace(path, match[0], f"{key} = {float(value)!r}   ")

    intercept = src / "INTERCEPT.F90"
    if unloading == 4:
        # CLM5 technical note 2.7.14-15; T is canopy temperature here.
        replace(intercept, "#elif CANUNL == 2", """#elif CANUNL == 4
    dsvg = (max(Tveg(n) - 270., 0.)/Tunl + Ua/Uunl)*dt*Sveg(n)
#elif CANUNL == 2""")
    if loading not in (3, 4):
        return

    # Supply air temperature to the canopy capacity calculation.
    for name in ("FSM2_TIMESTEP.F90", "CANOPY.F90"):
        replace(src / name, "CANOPY(Sveg,Tveg,VAI,", "CANOPY(Sveg,Tveg,Ta,VAI,")
    canopy = src / "CANOPY.F90"
    replace(canopy, "real, intent(in) :: &", "real, intent(in) :: Ta\nreal, intent(in) :: &")
    density = "67.92 + 51.25*exp((Ta-273.15)/2.59)"
    if loading == 4:
        density = f"min(120., {density})"
    replace(canopy, "Scap(:) = svai*lveg(:)", f"""! HP98 temperature-dependent capacity; Noah-MP caps density at 120.
  Scap(:) = svai*lveg(:)*(0.27 + 46./({density}))
  ! Capacity limits new loading. Warming alone cannot expel existing snow.
  Scap(:) = max(Scap(:), Sveg(:))""")

    intercept = src / "INTERCEPT.F90"
    if loading == 4:
        # Noah-MP CANWATER: full forest/precipitation coverage within the column.
        replace(intercept, "#elif CANINT == 2", """#elif CANINT == 4
    dsvg = max(0.,Scap(n)-Sveg(n)) * real(1.d0-exp(-dble(Sfg)*dt/Scap(n)))
    dsvg = min(dsvg, Sfg*dt)
#elif CANINT == 2""")
        return
    replace(intercept, "#elif CANINT == 2", """#elif CANINT == 3
    ! HP98 loading, dense-stand approximation Cp = 1.
    dsvg = max(0.,Scap(n)-Sveg(n)) * real(1.d0-exp(-dble(Sfg)*dt/Scap(n)))
    dsvg = min(dsvg, Sfg*dt)
#elif CANINT == 2""")
    replace(intercept, "#elif CANUNL == 2", """#elif CANUNL == 3
    ! Integrate dL/dt = -L/eunl over one model step.
    dsvg = Sveg(n)*real(1.d0-exp(-dble(dt)/eunl))
#elif CANUNL == 2""")


def build(source, output, factors=(1.,)):
    output.mkdir(parents=True, exist_ok=True)
    for name, loading, unloading, params in members(factors):
        final = output / name
        record = dict(
            name=name, commit=PIN, CANINT=loading, CANUNL=unloading,
            parameters=params,
            builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            shared_physics="FSM2 energy balance, sublimation and melt; ERA5 snowfall; VAI as L+S",
        )
        if (final / "FSM2").is_file():
            if json.loads((final / "member.json").read_text()) != record:
                raise ValueError(f"{final}: different build; choose a new output directory")
            continue
        print(f"Building {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            root = Path(tmp) / "source"
            subprocess.run(
                ["git", "clone", "-q", "--no-checkout", "--no-hardlinks",
                 str(source.resolve()), str(root)], check=True,
            )
            subprocess.run(["git", "checkout", "-q", "--detach", PIN], cwd=root, check=True)
            configure(root, loading, unloading, params)
            subprocess.run(["bash", "-eu", "compil.sh"], cwd=root, check=True)
            (root / "member.json").write_text(json.dumps(record, indent=2) + "\n")
            root.rename(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factors", type=float, nargs="+", default=[1.0],
                        help="Capacity and loss-rate multipliers, e.g. 0.8 1 1.2 (27 members)")
    args = parser.parse_args()
    if any(not 0 < f < float("inf") for f in args.factors):
        parser.error("Factors must be finite and positive")
    build(args.source, args.output, sorted(set(args.factors)))

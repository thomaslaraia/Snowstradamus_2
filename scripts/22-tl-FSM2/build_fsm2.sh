#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PIN=a76945e75b36628161f4679428ed72dce090b368
if [ ! -d FSM2_source ]; then
  git clone https://github.com/RichardEssery/FSM2.git FSM2_source
fi
cd FSM2_source
git checkout "$PIN"
python - <<'PY'
from pathlib import Path
p = Path('compil.sh').read_text()
for old, new in [('DRIV1D 1', 'DRIV1D 2'), ('SWPART 0', 'SWPART 1'), ('ZOFFST 0', 'ZOFFST 1')\
                 ('CANUNL 1', 'CANUNL 2')]:
    assert p.count(old) == 1, old
    p = p.replace(old, new)
Path('compile_era5.sh').write_text('set -eu\n' + p)
PY
bash compile_era5.sh
cp src/OPTS.h ../physics_options.h
git rev-parse HEAD > ../source_commit.txt
echo 'Built FSM2_source/FSM2'

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BETAS="0.01,0.286667,0.563333,0.84,1.11667,1.39333,1.67,1.94667,2.22333,2.5"
EQ_NL="1000,1000,1000,1000,1000,1000,2767,10773,41563,160000"
SW_NL="1000,1000,1000,1000,1000,1000,2767,10773,41563,160000"
EQ_L="100,100,100,170,493,1591,6923,36324,259751,2000001"
SW_L="200,200,200,339,985,3181,13846,72648,519502,4000001"
OUT=campaigns
mkdir -p "$OUT" plots
timeit() { /usr/bin/time -p "$@"; }
for backend in CPU GPU; do
  for mode in nonlocal local; do
    if [ "$mode" = nonlocal ]; then EQ=$EQ_NL; SW=$SW_NL; else EQ=$EQ_L; SW=$SW_L; fi
    tag=$(echo "$backend"_"$mode" | tr A-Z a-z)
    echo "=== $tag $(date -u +%FT%TZ) ==="
    timeit micromamba run -n rsmi python solid/aging_driver.py \
      --betas "$BETAS" --L 21 --rho 0.75 --rho1 0.30 --rho2 0.45 \
      --chains 8 --backend "$backend" --mode "$mode" \
      --equilibration "$EQ" --sweeps "$SW" \
      --output "$OUT/$tag.npz" --workdir "$OUT/$tag" \
      --driver solid/driver --max-parallel 1
    micromamba run -n rsmi python solid/plot_paper.py \
      --input "$OUT/$tag.npz" --output-dir "plots/$tag" --skip-fig2 || true
  done
done

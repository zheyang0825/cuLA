# Scripts

## Build

```bash
source .venv/bin/activate
pip install -e . --no-build-isolation
```

## Register / Resource Usage

```bash
cuobjdump -res-usage cula/cudac.cpython-312-x86_64-linux-gnu.so 2>&1 | grep -B1 -A3 "kda_bwd_intra"
```

## Benchmark

```bash
source .venv/bin/activate
python scripts/ncu_bwd_intra.py
```

## NCU Profiling (GUI report)

```bash
$(which ncu) --target-processes all \
  --launch-skip 1 --launch-count 1 --set full \
  -o bwd_intra \
  $(which python) scripts/ncu_bwd_intra.py
```

## Correctness Test

```bash
source .venv/bin/activate
python tests/test_kda_bwd_intra.py
```

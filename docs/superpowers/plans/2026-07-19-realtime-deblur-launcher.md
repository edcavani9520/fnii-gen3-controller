# Controller-Side Real-Time Deblur Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight Gen3 Controller entry point that delegates real-time RGB deblurring inference to the canonical sibling deblurring repository.

**Architecture:** A hardware-independent forwarding launcher parses only `--deblur-root`, dynamically loads the canonical launcher from that checkout, appends the current Gen3 repository as `--controller-root`, and forwards every other argument. Tests use a fake canonical checkout, while README instructions document the public repositories and direct controller-side command.

**Tech Stack:** Python 3, argparse, importlib, pathlib, pytest

---

## File Map

- Create `ws_inference_realtime_deblur.py`: controller-side path validation, deferred canonical-module loading, argument forwarding, and command-line entry point.
- Create `tests/test_ws_inference_realtime_deblur_launcher.py`: hardware-independent missing-checkout and forwarding tests.
- Modify `readme.md`: public clone URLs, sibling layout, dependencies, and controller-side invocation.

### Task 1: Deferred Forwarding Launcher

**Files:**
- Create: `ws_inference_realtime_deblur.py`
- Create: `tests/test_ws_inference_realtime_deblur_launcher.py`

- [ ] **Step 1: Write failing import-safety and missing-checkout tests**

Create a test that imports the controller-side module without loading Kortex or
the canonical deblur module, then calls `load_deblur_launcher(tmp_path)` and
asserts that `FileNotFoundError` mentions
`ws_inference_realtime_deblur.py`, the public deblurring URL, and `git clone`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_ws_inference_realtime_deblur_launcher.py -v
```

Expected: test collection fails because the controller-side launcher does not
exist.

- [ ] **Step 3: Implement deferred canonical loading**

Implement these interfaces in `ws_inference_realtime_deblur.py`:

```python
DEBLUR_REPOSITORY_URL = "https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git"
CONTROLLER_ROOT = Path(__file__).resolve().parent

def load_deblur_launcher(deblur_root):
    # Resolve <deblur_root>/ws_inference_realtime_deblur.py, validate it,
    # prepend deblur_root to sys.path, and load it with
    # importlib.util.spec_from_file_location("_canonical_realtime_deblur_ws", source).

def _parse_forwarding_args(argv=None):
    # Use ArgumentParser(add_help=False) with only --deblur-root and return
    # (known.deblur_root, remaining_arguments).
```

The default deblur root is
`../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin`. Do not import
NumPy, Kortex, WebSocket, OpenPI, or the canonical launcher at module import
time.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: import-safety and missing-checkout tests
pass.

- [ ] **Step 5: Write a failing forwarding test**

Create a temporary fake deblur checkout containing a canonical launcher whose
`main(argv)` records its argument list. Call controller-side `main()` with
`--deblur-root`, `--K 0.02`, and `--help`. Assert `--deblur-root` is consumed,
the other arguments preserve their order, and the final two arguments are:

```python
["--controller-root", str(controller_launcher.CONTROLLER_ROOT)]
```

- [ ] **Step 6: Implement exact forwarding**

Implement:

```python
def main(argv=None):
    deblur_root, forwarded = _parse_forwarding_args(argv)
    canonical = load_deblur_launcher(deblur_root)
    canonical.main([
        *forwarded,
        "--controller-root",
        str(CONTROLLER_ROOT),
    ])
```

Appending `--controller-root` last ensures the controller-side entry point
always uses its own checkout, even if a caller supplied an earlier value.

- [ ] **Step 7: Run all launcher tests and a syntax check**

Run:

```powershell
python -m compileall -q ws_inference_realtime_deblur.py
python -m pytest tests/test_ws_inference_realtime_deblur_launcher.py -v
```

Expected: compilation succeeds and all launcher tests pass without hardware
access.

- [ ] **Step 8: Commit the launcher and tests**

```powershell
git add -- ws_inference_realtime_deblur.py tests/test_ws_inference_realtime_deblur_launcher.py
git commit -m "feat: add controller-side real-time deblur launcher"
```

### Task 2: Public Checkout and Run Documentation

**Files:**
- Modify: `readme.md`

- [ ] **Step 1: Append the controller-side usage section**

Add a bounded `实时 RGB 去模糊 WS 推理` section containing both exact public
URLs, sibling clone commands, the `--deblur-root` behavior, runtime dependency
note, and this command:

```powershell
python ws_inference_realtime_deblur.py `
  --deblur-root ../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin `
  --ws-host localhost --ws-port 8000 `
  --K 0.01 --depth 0.5 --exposure 0.03
```

State explicitly that the canonical wrapper replaces policy
`observation/image` with the deblurred RGB frame.

- [ ] **Step 2: Verify documentation and canonical help delegation**

Run:

```powershell
rg -n "Robot-Kinematics|fnii-gen3-controller|observation/image|--deblur-root" readme.md
python ws_inference_realtime_deblur.py --deblur-root "E:\Vital_document\CUHKSZ\课程文件\ECE4512\Final" --help
```

Expected: README contains all four markers and the canonical help text prints.

- [ ] **Step 3: Run the Gen3 repository test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all available Gen3 Controller tests pass. If unrelated pre-existing
tests fail, preserve the exact output and separately re-run the focused
launcher test to establish feature status.

- [ ] **Step 4: Commit the documentation**

```powershell
git add -- readme.md
git commit -m "docs: explain controller-side real-time deblur launch"
```

- [ ] **Step 5: Review repository ownership and status**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: only intentional pre-existing changes, if any, remain; no deblurring
algorithm file has been copied into the Gen3 Controller repository.

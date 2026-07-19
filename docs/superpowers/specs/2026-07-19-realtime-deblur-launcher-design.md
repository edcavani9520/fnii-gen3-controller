# Gen3 Controller Real-Time Deblur Launcher Design

## Goal

Allow users to start the real-time RGB deblurring WebSocket inference workflow
from the `fnii-gen3-controller` checkout without duplicating or maintaining the
deblurring algorithms in this repository.

## Architecture

Add one lightweight `ws_inference_realtime_deblur.py` launcher to the Gen3
controller repository. The launcher accepts `--deblur-root`, whose default is
the expected sibling checkout
`../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin`. It validates
that the selected directory contains the canonical
`ws_inference_realtime_deblur.py`, adds the directory to `sys.path`, imports the
canonical module under an unambiguous module name, and delegates to its
`main()` function.

The launcher always supplies its own repository directory as
`--controller-root`. All remaining arguments are forwarded unchanged, so
Wiener and Pi05 controller options remain defined in one canonical parser.

## User Interface

The controller-side command is:

```powershell
python ws_inference_realtime_deblur.py `
  --deblur-root ../Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin `
  --ws-host localhost --ws-port 8000 --K 0.01 --depth 0.5 --exposure 0.03
```

The controller launcher owns only `--deblur-root`; `--help` and all other
arguments are forwarded to the canonical launcher. The Gen3 Controller README
will document both public GitHub URLs, the sibling directory layout, and this
command.

## Error Handling

If the canonical launcher cannot be found, fail before importing hardware
dependencies. The error identifies the expected file, shows the deblurring
repository URL, and provides its `git clone` command. Import failures from the
canonical launcher remain visible so missing runtime dependencies are not
masked.

## Testing

Add hardware-independent tests in the Gen3 repository using a temporary fake
deblurring checkout. Tests verify argument forwarding, automatic injection of
the local controller root, missing-checkout guidance, and import-time safety.
No test connects to the robot, camera, or WebSocket service.

## Ownership Boundary

The RGB Wiener implementation, kinematics, PSF generation, and wrapper subclass
remain exclusively in the deblurring repository. The Gen3 repository contains
only the forwarding launcher and documentation, preventing the two copies from
drifting apart.

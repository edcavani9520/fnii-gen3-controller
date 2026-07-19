import json
import sys

import pytest

import ws_inference_realtime_deblur as controller_launcher


def test_import_does_not_load_canonical_or_hardware_modules():
    assert "_canonical_realtime_deblur_ws" not in sys.modules
    assert "kortex_api.autogen.client_stubs.BaseClientRpc" not in sys.modules


def test_missing_deblur_checkout_has_public_clone_guidance(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        controller_launcher.load_deblur_launcher(tmp_path)

    message = str(exc_info.value)
    assert str(tmp_path / "ws_inference_realtime_deblur.py") in message
    assert controller_launcher.DEBLUR_REPOSITORY_URL in message
    assert "git clone https://" in message


def test_main_forwards_arguments_and_forces_local_controller_root(tmp_path):
    recorded_path = tmp_path / "forwarded.json"
    canonical_source = tmp_path / "ws_inference_realtime_deblur.py"
    canonical_source.write_text(
        "import json\n"
        f"OUTPUT = {str(recorded_path)!r}\n"
        "def main(argv=None):\n"
        "    with open(OUTPUT, 'w', encoding='utf-8') as stream:\n"
        "        json.dump(argv, stream)\n",
        encoding="utf-8",
    )

    controller_launcher.main(
        [
            "--deblur-root",
            str(tmp_path),
            "--K",
            "0.02",
            "--help",
            "--controller-root",
            "wrong-checkout",
        ]
    )

    forwarded = json.loads(recorded_path.read_text(encoding="utf-8"))
    assert "--deblur-root" not in forwarded
    assert forwarded[:5] == [
        "--K",
        "0.02",
        "--help",
        "--controller-root",
        "wrong-checkout",
    ]
    assert forwarded[-2:] == [
        "--controller-root",
        str(controller_launcher.CONTROLLER_ROOT),
    ]

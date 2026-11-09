"""Entry point:  python -m bowtrack.gui.app [--video FILE | --demo | --camera] [--model] [--checkpoint CKPT]"""

from __future__ import annotations

import argparse
import sys

from PySide6 import QtWidgets

from ..config import load_config
from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Violin bow-angle tracker GUI")
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--video", default=None, help="open a video file on start")
    ap.add_argument("--demo", action="store_true", help="start the synthetic demo")
    ap.add_argument("--camera", action="store_true", help="start the webcam on launch")
    ap.add_argument("--checkpoint", default=None, help="model checkpoint (.pt)")
    ap.add_argument("--model", action="store_true", help="tick 'Apply model' on start (run the keypoint model)")
    ap.add_argument("--no-sound", action="store_true", help="don't gate on sound / use the microphone")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("bowtrack")
    win = MainWindow(load_config(args.config), checkpoint=args.checkpoint)
    win.show()
    if args.no_sound:
        win.chk_sound.setChecked(False)
    if args.model:
        win.chk_model.setChecked(True)
    if args.video:
        win.open_video(args.video)
    elif args.demo:
        win.start_demo()
    elif args.camera:
        win.toggle_camera()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

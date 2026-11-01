#!/bin/sh
# Build the environment that scanning, calibration and meshing use (Open3D + OpenCV), on any distro.
#   ./setup-venv.sh                 .venv next to this script (the viewer looks for it there)
#   ./setup-venv.sh /tmp/try-venv   somewhere else, e.g. to try a newer Python first
#   PYTHON=python3.13 ./setup-venv.sh ...   with another interpreter
# An existing directory is left alone: remove it first to rebuild it.
set -e
cd "$(dirname "$0")"
dir=${1:-.venv}
py=${PYTHON:-python3}
if [ -e "$dir" ]; then
    echo "$dir exists; to rebuild it: rm -r $dir && $0 $*"
    exit 1
fi
"$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "needs Python 3.10 or newer"; exit 1; }
"$py" -c 'import tkinter' 2>/dev/null ||
    echo "note: $py has no tkinter; the viewer needs it (Debian/Ubuntu: python3-tk, Fedora: python3-tkinter)"
"$py" -m venv "$dir" ||
    { echo "could not make a venv (Debian/Ubuntu: install python3-venv)"; exit 1; }
"$dir/bin/python" -m pip install --quiet --upgrade pip
"$dir/bin/python" -m pip install --editable ".[scan]"
"$dir/bin/python" -c 'import numpy, open3d, cv2, sys; print("ok: python", sys.version.split()[0], "numpy", numpy.__version__,
    "open3d", open3d.__version__, "opencv", cv2.__version__)'

#!/bin/sh
# triBUUTSCAN live colour + IR + depth viewer (Creality Ferret Pro).
cd "$HOME/Desktop/dev/tribuutscan" && exec python3 -m tribuutscan.viewer "$@"

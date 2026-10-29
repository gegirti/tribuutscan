#!/bin/sh
# Live colour + IR + depth viewer for the Creality Ferret Pro.
cd "$HOME/Desktop/dev/ferret-linux" && exec python3 -m ferret.viewer "$@"

#!/usr/bin/env python3
"""Compatibility entry point for the corrected Cartographer overlay.

The former implementation performed an invalid room/map bounding-box
alignment.  Use ``overlay_marker_rays_on_cartographer_map.py`` instead; this
entry point deliberately forwards to that implementation so the old script
cannot silently recreate the incorrect output.
"""

from overlay_marker_rays_on_cartographer_map import main


if __name__ == "__main__":
    main()

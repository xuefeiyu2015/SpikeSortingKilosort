"""Probe geometry / channel maps.

Kilosort needs each channel's physical position. The two array types in this lab
could not be more different, and mixing them up silently produces plausible but
wrong sorting:

* **Neuropixels 1.0 NHP long** -- a dense linear/staggered shank, ~20 um pitch.
  Built from the run's own ``.meta`` (``~snsGeomMap``), which is authoritative
  per run and reflects the actual imro selection.
* **Blackrock Utah array** -- a 10x10 grid at 400 um pitch. Sites are far enough
  apart that no spike appears on two of them, so channels are effectively
  independent. Geometry comes from the array's ``.cmp`` map file.
"""

from .common import probe_summary, to_probeinterface  # noqa: F401
from .io import (  # noqa: F401
    load_probe_json,
    probe_from_mat,
    save_probe_json,
    save_probe_prb,
    validate_probe,
)
from .neuropixels import probe_from_geom, probe_from_meta  # noqa: F401
from .utah import independent_kcoords, probe_from_cmp, utah_grid_probe  # noqa: F401

"""Kilosort4 spike sorting pipeline for Blackrock and Neuropixels recordings.

Layout of the package mirrors the pipeline stages documented in ``CLAUDE.md``:

``config``   session + machine configuration (step 1)
``io``       readers for SpikeGLX and Blackrock files (steps 2, 4)
``sync``     sync pulse edge extraction and cross-system alignment (steps 2, 4, 8, 9)
``probes``   channel maps / probe geometry for each array type (step 3)
``sort``     Kilosort4 entry points (steps 3, 4)
``export``   post-curation metrics and final export (steps 7, 10)
``plots``    rendering only; never computes

Only ``numpy``/``scipy``/``pandas``/``yaml`` are imported at package import time.
Heavyweight, GPU-bound dependencies (``kilosort``, ``torch``, ``spikeinterface``,
``neo``) are imported lazily inside the modules that need them so that the config,
sync and export layers stay usable on a machine without a CUDA GPU.
"""

__version__ = "0.1.0"

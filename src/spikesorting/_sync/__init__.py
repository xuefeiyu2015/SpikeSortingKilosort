"""Synchronization pulse extraction and cross-system time alignment.

``edges``  pure NumPy edge detection (works everywhere)
``catgt``  wrapper around SpikeGLX's CatGT extractor (Windows/Linux only)
``extract`` drives both of the above from a SessionConfig
``burst``  14 s coded-burst matching -> coarse offset
``align``  overlap trimming, mapping, residual validation
``tprime`` wrapper around SpikeGLX's TPrime fine-alignment tool

Nothing in the sorting path may import ``catgt`` or ``tprime``: those tools are
absent on the HPC, and sorting must run there.
"""

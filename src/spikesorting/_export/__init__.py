"""Post-curation metrics and final export (pipeline steps 7 and 10).

``metrics``  pure computation on spike times and waveforms
``curated``  reads what Phy wrote back after manual curation
``final``    assembles the aligned, per-unit export

None of this needs Kilosort, torch or a GPU -- it reads the ``.npy``/``.tsv``
outputs directly, so curation and export can happen on any machine.
"""

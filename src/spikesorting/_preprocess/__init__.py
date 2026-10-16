"""Preprocessing before Kilosort: CatGT or SpikeInterface, chosen per session.

Kilosort then only whitens (and re-applies its own highpass). Which tool ran is
stated in the session's ``preprocess:`` block and recorded beside the binary it
wrote -- see :mod:`.provenance` -- so the session file and the output can never
disagree about what was done to the signal.

``catgt`` builds the CatGT command (and runs the sync extractors in the same
pass); ``si`` builds the SpikeInterface chain. Both are pure apart from the one
call each makes to run.
"""

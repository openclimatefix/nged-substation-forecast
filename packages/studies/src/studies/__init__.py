"""Tested machinery the one-off studies call.

A study under `studies/` answers a question once and is held to a lower standard than the rest of
the repository: no tests, no maintenance, no backwards compatibility. This package is the exception
inside that tier. Code lands here when a study got it wrong once, or when getting it wrong would be
silent — a centred rolling window off by one step, a stamp assigned to the wrong hour, a site label
derived two different ways — and everything here has tests.

`studies/README.md` states the split in full: what each study promises, and what it does not.
"""

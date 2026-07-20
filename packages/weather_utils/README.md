# weather_utils

Shared NWP query helpers used by more than one package, kept in a low-level package so that
neither consumer has to depend on the other.

Today this package owns the **NWP analysis-proxy query** (`select_analysis_proxy`): the closest
available proxy for the *true* weather at past times. We hold no weather observations, so the best
stand-in is the control ensemble member (member 0 — the unperturbed run started from the analysis
itself) at the shortest available lead: for each location and valid time, the freshest NWP run that
had been produced. Both the dashboard (which shows what a model sees) and the ML feature pipeline
(which builds weather-lag features) need exactly this selection, so it lives here and is computed
once.

# Studies

This section holds the **findings of experiments we have run**, written up so the conclusion
survives the code that produced it. It complements the other sections:
[background](../background/index.md) describes the *problem*, [techniques](../techniques/index.md)
explains the *methods*, the [roadmap](../roadmap/index.md) says what we plan to build, and
[architecture](../architecture/overview.md) documents what is already built.

**A page lands here when a study answers a question that outlives its code.** The code behind a page
here is deliberately lighter than production code — no Dagster asset, no data contract, and no
maintenance promise — and lives in
[`studies/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies),
which merges so that the measurement can be audited and re-run. What the project keeps is the
answer, which is why each page states its numbers, the checks the result survived, and the limits on
how far it generalises.

- [Does a weather product's beam/diffuse split help a PV forecast?](beam-diffuse-split.md) — on a 5
  km satellite retrieval the published direct-beam field cuts photovoltaic (PV) power error by 1.8%
  beyond what a separation model recovers from the total irradiance; on a 31 km reanalysis no effect
  is detected at the setting named before the run; and choosing the better irradiance product
  matters about 34 times more than the split does. The page also measures the two irradiance
  products against each other, a gradient-boosted tree against a fitted five-parameter physical PV
  model, what calibrating that physical model with a tree recovers, and what the per-site error
  drift says about estimating a generator's effective capacity. It reads NGED's two records of
  active network management against each other and against the telemetry, and finds the setpoint
  history the one to build on.

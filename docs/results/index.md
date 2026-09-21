# Results

This section holds the **findings of experiments we have run**, written up so the conclusion
survives the code that produced it. It complements the other sections:
[background](../background/index.md) describes the *problem*, [techniques](../techniques/index.md)
explains the *methods*, the [roadmap](../roadmap/index.md) says what we plan to build, and
[architecture](../architecture/overview.md) documents what is already built.

**A page lands here when an experiment answers a question that outlives its code.** Several of
these experiments are deliberately throwaway — no Dagster asset, no package, no data contract — and
their pull requests are closed without merging. What the project keeps is the answer, which is why
each page states its numbers, the checks the result survived, and the limits on how far it
generalises.

- [Does a weather product's beam/diffuse split help a PV forecast?](beam-diffuse-split.md) — on a
  5 km satellite retrieval the published direct-beam field cuts PV power error by 1.5% beyond what
  a separation model recovers from global irradiance; on a 31 km reanalysis it adds nothing; and
  choosing the better irradiance source matters about thirty times more than the split does. Also
  measures the two irradiance sources against each other, and a gradient-boosted tree against a
  fitted five-parameter physical PV model.

# Differentiable physics: pooling across substations

**The summary of this theme is in progress.** The differentiable-physics series builds models from
physical terms (demand, heating, solar) whose parameters are fitted by gradient descent. This theme
covers whether one such model fitted across many Primary substations can match a model per
substation, and beat it for a new substation or one with a short history. The
[Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home
message, and the [Research plan](plan.md) lists what is planned next. Experiment 112 found that
fitting the pooled model exactly does not remove its gap to per-substation models.

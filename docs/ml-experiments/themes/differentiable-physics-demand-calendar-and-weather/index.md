# Differentiable physics (demand, calendar, and weather)

**This theme covers the demand model of a six-part series on forecasting with physical terms.**

Background: [Differentiable physics: background](../differentiable-physics-background.md) explains
the models in plain terms.

The series builds models of Primary substation demand from a daily
profile, holiday and heating terms, and a solar term, fitted by gradient-based optimisation. This
theme holds the part that builds the model and asks whether more years of history help it. The
series' finding so far is in [experiment
104](../../00104-differentiable-physics-2-simple-structural-model/result.md): more years help a
convex model without drift, mainly through one substation. [Experiment
63](../../00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md) is an earlier, smaller linear
test of the same decomposition into a weather baseline, a solar term, and transfer terms. The
[Experiments](experiments.md) page lists every experiment, and the [Research plan](plan.md) lists
what is planned next.

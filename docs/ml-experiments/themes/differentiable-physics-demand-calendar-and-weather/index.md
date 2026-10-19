# Differentiable physics (demand, calendar, and weather)

**This theme covers the demand model of a six-part series on forecasting with physical terms.** The
series builds models of Primary substation demand from a daily profile, holiday and heating terms,
and a solar term, fitted by gradient-based optimisation. This theme holds the part that builds the
model and asks whether more years of history help it. The one finding so far is in [experiment
104](../../00104-differentiable-physics-2-simple-structural-model/result.md): the convex model beats
lagless XGBoost, and all years with drift are level with 12 months, so more years are not yet shown
to help. The [Experiments](experiments.md) page lists every experiment, and the [Research
plan](plan.md) lists what is planned next.

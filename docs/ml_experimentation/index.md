# ML Experimentation

How we run and evaluate ML forecasting experiments. Unlike the [roadmap](../roadmap/index.md)
(which holds forward-looking design for work not yet built), this area documents methodology that
is **implemented** and in use — the durable home for ML experimentation docs once they leave the
roadmap.

## Documents

- [Running an ML experiment end-to-end](dagster-workflow.md) — step-by-step recipe for going
  from raw data to a trained, MLflow-tracked model using the Dagster pipeline; explains why
  `trained_cv_model` reads config from MLflow rather than YAML.
- [Model configuration](model-configuration.md) — how to set hyperparameters and choose
  features; the full feature vocabulary and the lookahead-bias guardrails.
- [Cross-validation folds](cross-validation-folds.md) — the expanding-window CV protocol, the
  current single fold and why the data constrains us to it, the target multiple-yearly-fold
  protocol, and the fold-design alternatives we considered.

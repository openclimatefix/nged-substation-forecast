- In `ml_core/pyproject.toml`:
    - Can we remove the sections for `tool.ruff` and `tool.hatch` in `ml_core/pyproject.toml`? Can't
    these details be inherited from the `pyproject.toml` in the base of the uv workspace?
- In `ml_core/model.py`:
    - Please explain in the class docstring what the `Generic[T_InferReq]` is for. And please give a
      brief practical example of how an instance of this class is created and used.
    - in the docstring for `predict`, please explain why `context` and `params` are unused.
    (Actually, should we pass `context` and `params` through to `_run_inference`?
    - Please also explain in the docstring for `predict` why `predict` and `_run_inference` are both
      necessary: explain why we can't merge them.
    - in `predict`, should the type of `model_input` actually be `T_InferReq`?
    - Should `ForecastInference` also expose a `get_required_assets` method, just like
    `BaseTrainer`? If so, please add a `get_required_assets` method, and explain how this method
    should be called before calling `predict`.
    - Also, I know I suggested renaming the class to `ForecastInference`, but, now that I think more
      about it, I think I'd prefer to use a name for the class that is more consistent with
    `BaseTrainer`. Maybe `BaseInferenceModel` or something like that? (Please also change the name
    in `ml_core/README.md`).
- In `ml_core/trainer.py`:
    - It's not entirely clear to me what the first sentence in the docstring for
    `BaseDataRequirements` means? Why is the mapping _dynamic_? And also, in this docstring, should
    we mention that `BaseDataRequirements` is the base for the `T_TrainReq` and `T_InferReq`? Also,
    I wonder if we should move the paragraph about `ConfigDict` out of the docstring and into a code
    comment? (Do _users_ of the class need to know about `ConfigDict`?)
    - In the docstring for `get_required_assets`, instead of saying "we assume", should we say
    something more like "we check that every field name..."? (If I've understood correctly, casting to `FeatureAsset(f)` should fail if `f` isn't a member of `FeatureAsset`, right?)
    - Please add a sentence or two to the class docstring to explain what `Generic[T_TrainReq]`
    means.
    - Please add a brief explanation to the class docstring of how the `BaseTrainer` is meant to be used:
    i.e. the caller should first call `data_requirements()` to get a list of `FeatureAsset`s, and
    then call `train()` with the keys to `data` set to be the `FeatureAssets` returned by
    `data_requirements()`. `train()` returns an implementation of a `mlflow.pyfunc.PythonModel` that
    captures everything required to run the model (Don't use this exact text, please make my text a
    bit nicer!)
- In `xgboost_forecaster/model.py`:
    - Rename `XGBoostPolarsWrapper` to `

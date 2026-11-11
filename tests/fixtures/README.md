# Test fixtures

Small reference values used by the offline test suite. These are not model
artifacts, which is why `.gitignore` exempts this directory from the
repository's `*.safetensors` rule.

## `training_reference.{json,safetensors}`

Used by `tests/test_training_reference.py`, which checks the virtual-node
transform and the forward pass against values recorded from the training
implementation.

- **JSON** — the input OPF dictionary, demand statistics, metadata, input
  channels, model hyperparameters, expected totals, and the versions the
  values were generated with.
- **safetensors** — the augmented graph's tensors, the model weights, and the
  expected predictions for each `minmax_scaling` and demand-sum-rule setting.

The model is intentionally small: the fixture checks numerical agreement across
the full relation set, not capacity. The demand statistics are synthetic.

Regenerating these values is a manual, internal step.

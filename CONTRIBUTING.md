# Contributing

Contributions are welcome — bug reports, new benign transforms, sharper
report ideas, and tests.

## Development setup

```bash
git clone https://github.com/habib-analyst/probeshift.git
cd probeshift
python -m pip install -e ".[dev]"
python -m pytest
```

## Ground rules

- **NumPy is the only runtime dependency.** That is a product feature
  (framework-agnostic, callable-only); please do not add others.
- **Measurement, not attack.** ProbeShift sweeps fixed grids of benign
  transforms. No gradient methods, no optimisation toward a target
  class, no adaptive search — contributions that turn it into an
  attack generator are out of scope by design.
- **Deterministic or it doesn't ship.** Every probe run must be
  reproducible byte-for-byte from its inputs and recorded config. New
  transforms need fixed grids ordered by distance and identity
  parameters that return the input unchanged.
- **Every behaviour change needs a test.** The analytic tests in
  `tests/test_probeshift.py` are the contract: toy models with known
  decision boundaries must measure the hand-computed flip distance.
- **Honesty in numbers.** Examples print what they measure. If a change
  makes the demo numbers worse, the README and report say so — they are
  never tuned quietly.

## Reporting issues

Open a GitHub issue with a minimal reproduction: your `predict_fn`'s
output shape, your image batch shape/dtype, the ProbeShift version,
and the card you expected versus the card you got.

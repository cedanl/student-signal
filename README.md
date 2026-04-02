# student-signal

Domain library for student dropout prediction. Prepares student data, trains sklearn-compatible models, and ranks students by predicted dropout probability using the invitation rule methodology (Eegdeman, 2022).

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/cedanl/student-signal.git
cd student-signal
uv sync
```

## Usage

The library exposes three entry points: `prepare()`, `rank()`, and `evaluate()`.

```python
import pandas as pd
from student_signal.pipeline import prepare, rank, evaluate

train_df = pd.read_csv("data/01-raw/demo/synth_data_train.csv", sep="\t")
pred_df  = pd.read_csv("data/01-raw/demo/synth_data_pred.csv",  sep="\t")

# 1. Clean and transform
prepared = prepare(train_df, pred_df, target_col="Dropout", id_col="Studentnummer")

# 2. Train your model (any sklearn-compatible estimator)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(prepared.X_train, prepared.y_train)

# 3. Rank students by dropout risk
ranking = rank(model, prepared)

# 4. Evaluate with stoplight methodology
results = evaluate({"Random Forest": (model, False)}, prepared, invite_pct=20)
```

`rank()` returns a DataFrame with columns `ranking`, `Studentnummer`, and `voorspelling`, sorted highest risk first.

`evaluate()` returns precision/recall metrics at multiple invitation thresholds with a traffic light status (Betrouwbaar / Gebruik met voorzichtigheid / Niet bruikbaar).

## Development

Open in devcontainer or run locally:

```bash
uv sync
uv run pytest          # run tests
uv run ruff check src  # lint
```

## License

MIT

"""student-signal: dropout prediction library for student intervention planning.

Public API
----------
prepare(train_df, pred_df, ...)  →  PreparedData
rank(model, prepared, ...)       →  DataFrame (students ranked by dropout risk)
evaluate(models, prepared, ...)  →  dict (stoplight metrics per model)

Lower-level building blocks are available via submodules for consumers
that need more control:

    from student_signal.features import standardize_dataset
    from student_signal.modeling.train import train_random_forest
    from student_signal.evaluate import load_settings, generate_stoplight_evaluation
"""

from student_signal.pipeline import PreparedData, evaluate, prepare, rank

__all__ = [
    "PreparedData",
    "prepare",
    "rank",
    "evaluate",
]

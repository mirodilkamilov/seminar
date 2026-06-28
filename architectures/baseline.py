from architectures.architecture import Architecture


class Baseline(Architecture):
    """
    Native function calling, no reasoning scaffold — the control condition.
    Its system prompt is exactly the shared ``BASE_TASK_INSTRUCTION``
    (``reasoning_scaffold`` left empty), so every other architecture differs
    from it by reasoning/reflection text alone.
    """

    name = "baseline"

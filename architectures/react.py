from architectures.architecture import Architecture


class ReAct(Architecture):
    """
    ReAct-style prompting: the model states a brief Thought (free text) before
    acting. Actions still go through the native FC channel — we test the
    *thought injection*, not a text action format. Zero-shot (no exemplars).
    """

    name = "react"

    # Only the reasoning scaffold differs from the baseline; the task core and
    # turn-completion protocol live in BASE_TASK_INSTRUCTION (shared by all).
    # NB: reasoning is forward (Thought before each action).
    reasoning_scaffold = (
        "Work in an explicit reason-act loop. Every time you call a tool, begin "
        "that SAME message with the literal label 'Thought:' followed by one or "
        "two sentences — what you know, what you still need, and which tool to "
        "call next — then make the tool call(s) in that same message.\n"
        "A Thought with no tool call attached ends the current turn. "
        "After each tool result, any further tool call again begins with a "
        "'Thought:' label in that same message."
    )

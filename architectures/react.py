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
        "Work in an explicit reason-act loop. Every time you call a tool, first "
        "write a short 'Thought:' in the SAME message — what you know, what you "
        "still need, and which tool to call next — then make the tool call(s) in "
        "that same message. Never send a Thought with no tool call attached. "
        "After each tool result, your next message again begins with a brief "
        "Thought followed by the next tool call."
    )

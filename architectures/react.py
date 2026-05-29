from architectures.architecture import Architecture


class ReAct(Architecture):
    """
    ReAct-style prompting: the model states a brief Thought (free text) before
    acting. Actions still go through the native FC channel — we test the
    *thought injection*, not a text action format. Zero-shot (no exemplars).
    """

    name = "react"

    def system_prompt(self) -> str:
        return (
            "You are an agent that completes user tasks by calling the provided "
            "tools. Work in an explicit reason-act loop: before each tool call, "
            "briefly state your reasoning as a short 'Thought:' in your message "
            "content — what you know, what you still need, and which tool to call "
            "next — then call the appropriate tool(s). After observing the tool "
            "results, reason again before the next action. When you have "
            "everything needed to satisfy the user's request, reply in natural "
            "language to summarise what you did."
        )

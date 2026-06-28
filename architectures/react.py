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
            "tools. Work in an explicit reason-act loop. Every time you call a "
            "tool, first write a short 'Thought:' in the SAME message — what you "
            "know, what you still need, and which tool to call next — then make "
            "the tool call(s) in that same message. Never send a Thought with no "
            "tool call attached. After each tool result, your next message again "
            "begins with a brief Thought followed by the next tool call. Only when "
            "the user's request is fully satisfied, reply in natural language with "
            "no tool call, summarising what you did."
        )

    # TODO: Do reasoning after the act?

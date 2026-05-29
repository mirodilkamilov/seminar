from architectures.architecture import Architecture


class Baseline(Architecture):
    """Native function calling, no reasoning prompt — the control condition."""

    name = "baseline"

    def system_prompt(self) -> str:
        return (
            "You are an agent that completes user tasks by calling the provided "
            "tools. Call tools whenever they are needed. After you have all the "
            "information to satisfy the user's request, reply in natural language "
            "to summarise what you did."
        )

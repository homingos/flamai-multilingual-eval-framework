from src.core.domain.model_config import ModelState

# Valid transitions — deprecated is terminal (no way out)
TRANSITIONS: dict[ModelState, set[ModelState]] = {
    ModelState.ACTIVE:     {ModelState.DISABLED, ModelState.DEPRECATED},
    ModelState.DISABLED:   {ModelState.ACTIVE,   ModelState.DEPRECATED},
    ModelState.DEPRECATED: set(),
}


def validate_transition(current: ModelState, target: ModelState) -> None:
    """Raise ValueError with a clear message if the transition is invalid."""
    allowed = TRANSITIONS[current]
    if target not in allowed:
        allowed_labels = [s.value for s in allowed] or ["none (terminal state)"]
        raise ValueError(
            f"Cannot transition model from '{current.value}' to '{target.value}'. "
            f"Valid transitions from '{current.value}': {allowed_labels}."
        )

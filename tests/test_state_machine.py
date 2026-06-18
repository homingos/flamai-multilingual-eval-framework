import pytest

from src.core.domain.model_config import ModelState
from src.core.services.state_machine import validate_transition


def test_active_to_disabled_allowed():
    validate_transition(ModelState.ACTIVE, ModelState.DISABLED)  # no raise


def test_disabled_to_active_allowed():
    validate_transition(ModelState.DISABLED, ModelState.ACTIVE)


def test_active_to_deprecated_allowed():
    validate_transition(ModelState.ACTIVE, ModelState.DEPRECATED)


def test_disabled_to_deprecated_allowed():
    validate_transition(ModelState.DISABLED, ModelState.DEPRECATED)


def test_deprecated_to_active_blocked():
    with pytest.raises(ValueError):
        validate_transition(ModelState.DEPRECATED, ModelState.ACTIVE)


def test_deprecated_to_disabled_blocked():
    with pytest.raises(ValueError):
        validate_transition(ModelState.DEPRECATED, ModelState.DISABLED)


def test_deprecated_to_deprecated_blocked():
    with pytest.raises(ValueError):
        validate_transition(ModelState.DEPRECATED, ModelState.DEPRECATED)


def test_active_to_active_blocked():
    with pytest.raises(ValueError):
        validate_transition(ModelState.ACTIVE, ModelState.ACTIVE)


def test_invalid_transition_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        validate_transition(ModelState.DEPRECATED, ModelState.ACTIVE)
    assert "deprecated" in str(exc_info.value).lower()
    assert "active" in str(exc_info.value).lower()


def test_error_message_names_current_and_target_state():
    with pytest.raises(ValueError) as exc_info:
        validate_transition(ModelState.DEPRECATED, ModelState.DISABLED)
    msg = str(exc_info.value)
    assert "deprecated" in msg
    assert "disabled" in msg
    assert "terminal" in msg or "none" in msg.lower()

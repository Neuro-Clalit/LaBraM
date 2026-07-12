"""Shared pytest fixtures.

``configure_logging`` intentionally mutates the process-global ``labram`` logger
(attaching handlers and setting ``propagate = False`` so training output is not
double-logged through the root logger). Tests that call it — or that assert on
records via ``caplog`` — must not leak that state into later tests: pytest's
``caplog`` captures through a handler on the root logger reached by propagation,
so a lingering ``propagate = False`` silently drops every ``labram`` record and
makes downstream tests fail depending only on collection order.

The autouse fixture below snapshots and restores the logger's handlers, level,
and propagation around every test, making the suite order-independent.
"""
import logging

import pytest

from labram.utils.logging import LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_labram_logger():
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    try:
        yield
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate

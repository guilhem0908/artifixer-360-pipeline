# SPDX-FileCopyrightText: Copyright (c) 2026 Guilhem Carmouze.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration."""

import pytest

# ERROR_PRIVILEGE_NOT_HELD: raised by os.symlink on Windows accounts without
# Developer Mode or elevation. Several pipeline steps link frames instead of
# copying them, so their tests cannot run there; they do run on Linux.
_WINDOWS_SYMLINK_PRIVILEGE_ERROR = 1314


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    error = call.excinfo.value if call.excinfo is not None else None
    if (
        report.when == "call"
        and isinstance(error, OSError)
        and getattr(error, "winerror", None) == _WINDOWS_SYMLINK_PRIVILEGE_ERROR
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] or 0,
            "Skipped: creating symbolic links needs Developer Mode or elevation on Windows",
        )

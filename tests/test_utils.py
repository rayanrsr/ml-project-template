"""Tests for the utils (file locking and friends)."""

from pathlib import Path

from src.utils.utils import file_lock_operation


def test_file_lock_operation_uses_a_shared_lock_file() -> None:
    """All processes must contend on the *same* lock file.

    Regression test: the lock file used to be created in a `tempfile.TemporaryDirectory()`, which
    is unique per call, so every process locked its own private file and there was no mutual
    exclusion at all.
    """
    seen: list[Path] = []

    def operation(path: Path) -> str:
        seen.append(path)
        return "done"

    assert file_lock_operation("project-template-test.lock", operation) == "done"
    assert file_lock_operation("project-template-test.lock", operation) == "done"

    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert seen[0].name == "project-template-test.lock"

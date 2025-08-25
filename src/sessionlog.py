"""
Utilities for writing results to disk.
"""

from pathlib import Path


class ResultsDir:
    """
    Overall results directory, containing subdirs for each session.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def get_session_ids(self):
        return [p.name for p in self.path.iterdir() if p.is_dir()]

    def get_session(self, id: str):
        return SessionDir(self.path / id)


class SessionDir:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def is_done(self):
        return (self.path / "done.txt").exists()

"""Append-only JSONL events with a checkpointed byte/hash cursor."""

import hashlib
import json
import os
from pathlib import Path


class TrainingJournal:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise ValueError("Training journal is not empty; restore its checkpoint cursor")
        self.path.touch()
        self.offset, self.digest = 0, hashlib.sha256()

    def append(self, record):
        payload = (json.dumps(record, allow_nan=False) + "\n").encode()
        with self.path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.offset += len(payload)
        self.digest.update(payload)

    def state_dict(self):
        return {"path": str(self.path), "bytes": self.offset, "sha256": self.digest.hexdigest()}

    @classmethod
    def restore(cls, state, target=None, *, fallback=None):
        destination = Path(target or state["path"]).resolve()
        source = Path(state["path"])
        if not source.exists() and fallback is not None:
            source = Path(fallback)
        if not source.exists() or source.stat().st_size < state["bytes"]:
            raise ValueError("Training journal is missing or shorter than its checkpoint cursor")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        digest, remaining = hashlib.sha256(), state["bytes"]
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                while remaining:
                    chunk = reader.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Training journal prefix is incomplete")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    writer.write(chunk)
                if digest.hexdigest() != state["sha256"]:
                    raise ValueError("Training journal prefix hash does not match checkpoint")
                if source.resolve() == destination and source.stat().st_size > state["bytes"]:
                    tail = destination.with_name(destination.stem + ".uncommitted.jsonl")
                    with tail.open("wb") as backup:
                        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                            backup.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        journal = cls.__new__(cls)
        journal.path, journal.offset, journal.digest = destination, state["bytes"], digest
        return journal

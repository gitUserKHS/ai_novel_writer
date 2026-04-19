from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .models import (
    ArtifactOut,
    BibleContent,
    JobOut,
    JobStatus,
    OutlineCard,
    PoolType,
    SceneOut,
    StoryCreate,
    StoryOut,
    StoryState,
    StoryUpdate,
    utcnow_iso,
)
from .utils.text import normalize_list, slugify


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS stories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    premise TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    themes_json TEXT NOT NULL,
                    characters_json TEXT NOT NULL,
                    forbidden_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    target_length_scenes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS story_bibles (
                    story_id TEXT PRIMARY KEY,
                    content_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS outline_cards (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    scene_order INTEGER NOT NULL,
                    card_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_outline_story ON outline_cards(story_id, scene_order);
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    scene_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    pov TEXT NOT NULL,
                    location TEXT NOT NULL,
                    time_label TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    accepted_text TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    extraction_json TEXT NOT NULL,
                    consistency_json TEXT NOT NULL,
                    creativity_json TEXT NOT NULL,
                    revision_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_scenes_story ON scenes(story_id, scene_index);
                CREATE TABLE IF NOT EXISTS scene_candidates (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    score REAL NOT NULL,
                    accepted INTEGER NOT NULL,
                    consistency_json TEXT NOT NULL,
                    creativity_json TEXT NOT NULL,
                    revision_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE,
                    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_scene ON scene_candidates(scene_id, candidate_index);
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT,
                    scene_index INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_state_story ON state_snapshots(story_id, scene_index);
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE,
                    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_kg_story ON kg_edges(story_id);
                CREATE TABLE IF NOT EXISTS dataset_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT,
                    pool_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_story ON dataset_records(story_id, pool_type);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    logs_json TEXT NOT NULL,
                    result_json TEXT,
                    error_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_story ON artifacts(story_id, created_at DESC);
                """
            )
            conn.commit()

    @staticmethod
    def _loads_json(value: str | None, default: Any) -> Any:
        if not value:
            return default
        return json.loads(value)

    @staticmethod
    def _dumps_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    def list_stories(self) -> List[StoryOut]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM stories ORDER BY updated_at DESC, created_at DESC").fetchall()
        return [self._story_row_to_model(row) for row in rows]

    def create_story(self, payload: StoryCreate) -> StoryOut:
        story_id_base = slugify(payload.title)
        story_id = story_id_base
        suffix = 1
        while self.get_story(story_id) is not None:
            suffix += 1
            story_id = f"{story_id_base}-{suffix}"
        now = utcnow_iso()
        record = {
            "id": story_id,
            "title": payload.title,
            "genre": payload.genre,
            "premise": payload.premise,
            "tone": payload.tone,
            "themes_json": self._dumps_json(normalize_list(payload.themes)),
            "characters_json": self._dumps_json(normalize_list(payload.characters)),
            "forbidden_json": self._dumps_json(normalize_list(payload.forbidden_facts)),
            "notes": payload.notes,
            "target_length_scenes": payload.target_length_scenes,
            "created_at": now,
            "updated_at": now,
        }
        default_bible = BibleContent(
            static_facts=[payload.premise],
            rules=normalize_list(payload.forbidden_facts),
            voice_notes=[payload.tone],
            motifs=normalize_list(payload.themes),
            reference_snippets=normalize_list(payload.characters),
        )
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO stories (
                    id, title, genre, premise, tone, themes_json, characters_json,
                    forbidden_json, notes, target_length_scenes, created_at, updated_at
                ) VALUES (:id, :title, :genre, :premise, :tone, :themes_json, :characters_json,
                          :forbidden_json, :notes, :target_length_scenes, :created_at, :updated_at)
                """,
                record,
            )
            conn.execute(
                "INSERT INTO story_bibles (story_id, content_json, updated_at) VALUES (?, ?, ?)",
                (story_id, self._dumps_json(default_bible.model_dump()), now),
            )
            conn.execute(
                "INSERT INTO state_snapshots (story_id, scene_id, scene_index, state_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, None, 0, self._dumps_json(StoryState().model_dump()), now),
            )
            conn.commit()
        return self.get_story(story_id)

    def get_story(self, story_id: str) -> Optional[StoryOut]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        return self._story_row_to_model(row) if row else None

    def update_story(self, story_id: str, payload: StoryUpdate) -> Optional[StoryOut]:
        current = self.get_story(story_id)
        if current is None:
            return None
        data = current.model_dump()
        for key, value in payload.model_dump(exclude_unset=True).items():
            if value is None:
                continue
            data[key] = value
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE stories SET
                    title = ?, genre = ?, premise = ?, tone = ?, themes_json = ?, characters_json = ?,
                    forbidden_json = ?, notes = ?, target_length_scenes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    data["title"],
                    data["genre"],
                    data["premise"],
                    data["tone"],
                    self._dumps_json(normalize_list(data["themes"])),
                    self._dumps_json(normalize_list(data["characters"])),
                    self._dumps_json(normalize_list(data["forbidden_facts"])),
                    data["notes"],
                    data["target_length_scenes"],
                    now,
                    story_id,
                ),
            )
            conn.commit()
        return self.get_story(story_id)

    def _story_row_to_model(self, row: sqlite3.Row | None) -> StoryOut:
        assert row is not None
        return StoryOut(
            id=row["id"],
            title=row["title"],
            genre=row["genre"],
            premise=row["premise"],
            tone=row["tone"],
            themes=self._loads_json(row["themes_json"], []),
            characters=self._loads_json(row["characters_json"], []),
            forbidden_facts=self._loads_json(row["forbidden_json"], []),
            notes=row["notes"],
            target_length_scenes=row["target_length_scenes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_bible(self, story_id: str) -> BibleContent:
        with self.connect() as conn:
            row = conn.execute("SELECT content_json FROM story_bibles WHERE story_id = ?", (story_id,)).fetchone()
        if row is None:
            return BibleContent()
        return BibleContent.model_validate(self._loads_json(row[0], {}))

    def save_bible(self, story_id: str, bible: BibleContent) -> BibleContent:
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO story_bibles (story_id, content_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(story_id) DO UPDATE SET content_json = excluded.content_json, updated_at = excluded.updated_at",
                (story_id, self._dumps_json(bible.model_dump()), now),
            )
            conn.execute("UPDATE stories SET updated_at = ? WHERE id = ?", (now, story_id))
            conn.commit()
        return bible

    def get_latest_state(self, story_id: str) -> StoryState:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM state_snapshots WHERE story_id = ? ORDER BY scene_index DESC, id DESC LIMIT 1",
                (story_id,),
            ).fetchone()
        if row is None:
            return StoryState()
        return StoryState.model_validate(self._loads_json(row[0], {}))

    def save_state_snapshot(self, story_id: str, scene_id: Optional[str], scene_index: int, state: StoryState) -> StoryState:
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO state_snapshots (story_id, scene_id, scene_index, state_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, scene_id, scene_index, self._dumps_json(state.model_dump()), now),
            )
            conn.execute("UPDATE stories SET updated_at = ? WHERE id = ?", (now, story_id))
            conn.commit()
        return state

    def get_next_scene_index(self, story_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(scene_index), 0) FROM scenes WHERE story_id = ?", (story_id,)).fetchone()
        return int(row[0]) + 1

    def save_scene(self, scene_payload: Dict[str, Any], candidates: List[Dict[str, Any]]) -> SceneOut:
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO scenes (
                    id, story_id, scene_index, title, pov, location, time_label, goal,
                    input_json, plan_json, accepted_text, summary, extraction_json,
                    consistency_json, creativity_json, revision_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_payload["id"],
                    scene_payload["story_id"],
                    scene_payload["scene_index"],
                    scene_payload["title"],
                    scene_payload["pov"],
                    scene_payload["location"],
                    scene_payload["time_label"],
                    scene_payload["goal"],
                    self._dumps_json(scene_payload["input"]),
                    self._dumps_json(scene_payload["plan"]),
                    scene_payload["accepted_text"],
                    scene_payload["summary"],
                    self._dumps_json(scene_payload["extraction"]),
                    self._dumps_json(scene_payload["consistency"]),
                    self._dumps_json(scene_payload["creativity"]),
                    self._dumps_json(scene_payload.get("revision")) if scene_payload.get("revision") is not None else None,
                    now,
                ),
            )
            for cand in candidates:
                conn.execute(
                    """
                    INSERT INTO scene_candidates (
                        id, story_id, scene_id, candidate_index, text, score, accepted,
                        consistency_json, creativity_json, revision_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cand["id"],
                        scene_payload["story_id"],
                        scene_payload["id"],
                        cand["candidate_index"],
                        cand["text"],
                        cand["score"],
                        1 if cand.get("accepted") else 0,
                        self._dumps_json(cand["consistency"]),
                        self._dumps_json(cand["creativity"]),
                        self._dumps_json(cand.get("revision")) if cand.get("revision") is not None else None,
                        now,
                    ),
                )
            for edge in scene_payload["extraction"].get("kg_edges", []):
                conn.execute(
                    "INSERT INTO kg_edges (story_id, scene_id, source, relation, target, edge_type, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scene_payload["story_id"],
                        scene_payload["id"],
                        edge["source"],
                        edge["relation"],
                        edge["target"],
                        edge.get("edge_type", "event"),
                        self._dumps_json(edge.get("metadata", {})),
                        now,
                    ),
                )
            conn.execute("UPDATE stories SET updated_at = ? WHERE id = ?", (now, scene_payload["story_id"]))
            conn.commit()
        return self.get_scene(scene_payload["story_id"], scene_payload["id"])

    def get_scene(self, story_id: str, scene_id: str) -> Optional[SceneOut]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM scenes WHERE story_id = ? AND id = ?", (story_id, scene_id)).fetchone()
            if row is None:
                return None
            cand_rows = conn.execute(
                "SELECT * FROM scene_candidates WHERE scene_id = ? ORDER BY candidate_index ASC",
                (scene_id,),
            ).fetchall()
        return self._scene_row_to_model(row, cand_rows)

    def list_scenes(self, story_id: str) -> List[SceneOut]:
        with self.connect() as conn:
            scene_rows = conn.execute(
                "SELECT * FROM scenes WHERE story_id = ? ORDER BY scene_index ASC",
                (story_id,),
            ).fetchall()
            out: List[SceneOut] = []
            for row in scene_rows:
                cand_rows = conn.execute(
                    "SELECT * FROM scene_candidates WHERE scene_id = ? ORDER BY candidate_index ASC",
                    (row["id"],),
                ).fetchall()
                out.append(self._scene_row_to_model(row, cand_rows))
        return out

    def _scene_row_to_model(self, row: sqlite3.Row, cand_rows: Iterable[sqlite3.Row]) -> SceneOut:
        return SceneOut(
            id=row["id"],
            scene_index=row["scene_index"],
            title=row["title"],
            pov=row["pov"],
            location=row["location"],
            time_label=row["time_label"],
            goal=row["goal"],
            accepted_text=row["accepted_text"],
            summary=row["summary"],
            created_at=row["created_at"],
            plan=self._loads_json(row["plan_json"], {}),
            extraction=self._loads_json(row["extraction_json"], {}),
            consistency=self._loads_json(row["consistency_json"], {}),
            creativity=self._loads_json(row["creativity_json"], {}),
            revision=self._loads_json(row["revision_json"], None),
            candidates=[
                {
                    "id": cand["id"],
                    "candidate_index": cand["candidate_index"],
                    "text": cand["text"],
                    "score": cand["score"],
                    "accepted": bool(cand["accepted"]),
                    "consistency": self._loads_json(cand["consistency_json"], {}),
                    "creativity": self._loads_json(cand["creativity_json"], {}),
                    "revision": self._loads_json(cand["revision_json"], None),
                    "created_at": cand["created_at"],
                }
                for cand in cand_rows
            ],
        )

    def list_kg_edges(self, story_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_edges WHERE story_id = ? ORDER BY id ASC",
                (story_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "scene_id": row["scene_id"],
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "edge_type": row["edge_type"],
                "metadata": self._loads_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_dataset_record(self, story_id: str, scene_id: Optional[str], pool_type: PoolType, payload: Dict[str, Any]) -> None:
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO dataset_records (story_id, scene_id, pool_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, scene_id, pool_type.value, self._dumps_json(payload), utcnow_iso()),
            )
            conn.commit()

    def dataset_counts(self, story_id: str) -> Dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT pool_type, COUNT(*) AS cnt FROM dataset_records WHERE story_id = ? GROUP BY pool_type",
                (story_id,),
            ).fetchall()
        counts = {pool.value: 0 for pool in PoolType}
        for row in rows:
            counts[row["pool_type"]] = row["cnt"]
        return counts

    def create_job(self, job_id: str, story_id: str, kind: str, message: str = "Queued") -> JobOut:
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, story_id, kind, status, progress, message, logs_json, result_json, error_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, story_id, kind, JobStatus.QUEUED.value, 0.0, message, self._dumps_json([]), None, "", now, now),
            )
            conn.commit()
        return self.get_job(job_id)

    def recover_incomplete_jobs(self) -> int:
        updated_at = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, message = ?, error_text = CASE
                    WHEN error_text = '' THEN ?
                    ELSE error_text
                END, updated_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    1.0,
                    "Interrupted",
                    "Job was interrupted by process shutdown or restart.",
                    updated_at,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
        return int(cursor.rowcount or 0)

    def append_job_log(self, job_id: str, message: str, progress: Optional[float] = None, status: Optional[JobStatus] = None) -> JobOut:
        job = self.get_job(job_id)
        assert job is not None
        logs = list(job.logs)
        logs.append({"time": utcnow_iso(), "message": message})
        progress_value = job.progress if progress is None else float(progress)
        status_value = job.status.value if status is None else status.value
        updated_at = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET logs_json = ?, progress = ?, status = ?, message = ?, updated_at = ? WHERE id = ?",
                (self._dumps_json(logs), progress_value, status_value, message, updated_at, job_id),
            )
            conn.commit()
        return self.get_job(job_id)

    def finish_job(self, job_id: str, result: Dict[str, Any], message: str = "Completed") -> JobOut:
        updated_at = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, message = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (JobStatus.SUCCEEDED.value, 1.0, message, self._dumps_json(result), updated_at, job_id),
            )
            conn.commit()
        return self.get_job(job_id)

    def fail_job(self, job_id: str, error_text: str) -> JobOut:
        updated_at = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, message = ?, error_text = ?, updated_at = ? WHERE id = ?",
                (JobStatus.FAILED.value, 1.0, "Failed", error_text, updated_at, job_id),
            )
            conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[JobOut]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobOut(
            id=row["id"],
            story_id=row["story_id"],
            kind=row["kind"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            message=row["message"],
            logs=self._loads_json(row["logs_json"], []),
            result=self._loads_json(row["result_json"], None),
            error_text=row["error_text"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_artifact(self, story_id: str, artifact_type: str, path: str, metadata: Dict[str, Any]) -> ArtifactOut:
        created_at = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO artifacts (story_id, artifact_type, path, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, artifact_type, path, self._dumps_json(metadata), created_at),
            )
            conn.commit()
            artifact_id = int(cursor.lastrowid)
        return ArtifactOut(id=artifact_id, artifact_type=artifact_type, path=path, metadata=metadata, created_at=created_at)

    def list_artifacts(self, story_id: str) -> List[ArtifactOut]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE story_id = ? ORDER BY id DESC",
                (story_id,),
            ).fetchall()
        return [
            ArtifactOut(
                id=row["id"],
                artifact_type=row["artifact_type"],
                path=row["path"],
                metadata=self._loads_json(row["metadata_json"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_artifact(self, story_id: str, artifact_id: int) -> Optional[ArtifactOut]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE story_id = ? AND id = ?",
                (story_id, artifact_id),
            ).fetchone()
        if row is None:
            return None
        return ArtifactOut(
            id=row["id"],
            artifact_type=row["artifact_type"],
            path=row["path"],
            metadata=self._loads_json(row["metadata_json"], {}),
            created_at=row["created_at"],
        )

    def list_outline(self, story_id: str) -> List[OutlineCard]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM outline_cards WHERE story_id = ? ORDER BY scene_order ASC",
                (story_id,),
            ).fetchall()
        cards: List[OutlineCard] = []
        for row in rows:
            payload = self._loads_json(row["card_json"], {})
            payload["id"] = row["id"]
            payload["status"] = row["status"]
            cards.append(OutlineCard.model_validate(payload))
        return cards

    def replace_outline(self, story_id: str, cards: List[OutlineCard]) -> List[OutlineCard]:
        now = utcnow_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute("DELETE FROM outline_cards WHERE story_id = ?", (story_id,))
            for idx, card in enumerate(cards, start=1):
                # Provider-generated outline ids are often reused across stories
                # (for example, "outline-01"), so persist story-scoped ids only.
                card_id = f"{story_id}-outline-{idx:02d}"
                payload = card.model_dump(exclude={"id", "status"})
                conn.execute(
                    "INSERT INTO outline_cards (id, story_id, scene_order, card_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (card_id, story_id, idx, self._dumps_json(payload), card.status, now, now),
                )
            conn.execute("UPDATE stories SET updated_at = ? WHERE id = ?", (now, story_id))
            conn.commit()
        return self.list_outline(story_id)

    def mark_outline_used(self, story_id: str, outline_card_id: str) -> None:
        with self._write_lock, self.connect() as conn:
            conn.execute(
                "UPDATE outline_cards SET status = ?, updated_at = ? WHERE story_id = ? AND id = ?",
                ("used", utcnow_iso(), story_id, outline_card_id),
            )
            conn.commit()

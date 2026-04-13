
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ArtifactRecord, BibleContent, OutlineCard, PoolType, StoryCreate, StoryMeta, StoryState, StoryUpdate, utcnow_iso
from .utils import ensure_dir, normalize_list, slugify


class Storage:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        ensure_dir(Path(self.database_path).parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS stories (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bibles (
                    story_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outline_cards (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    scene_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_outline_story_scene ON outline_cards (story_id, scene_index);

                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    scene_index INTEGER NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scenes_story_scene ON scenes (story_id, scene_index);

                CREATE TABLE IF NOT EXISTS scene_candidates (
                    id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scene_candidates_scene ON scene_candidates (scene_id);

                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT,
                    scene_index INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_story_scene ON state_snapshots (story_id, scene_index);

                CREATE TABLE IF NOT EXISTS kg_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kg_story ON kg_edges (story_id);

                CREATE TABLE IF NOT EXISTS dataset_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    scene_id TEXT,
                    pool_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_story_pool ON dataset_records (story_id, pool_type);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    story_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_story ON artifacts (story_id, created_at);
                """
            )

    @staticmethod
    def _dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _load(raw: str | None) -> Any:
        return json.loads(raw) if raw else None

    def create_story(self, payload: StoryCreate) -> StoryMeta:
        story_id = payload.id or slugify(payload.title)
        now = utcnow_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT 1 FROM stories WHERE id = ?", (story_id,)).fetchone()
            if existing is not None:
                suffix = uuid.uuid4().hex[:6]
                story_id = f"{story_id}-{suffix}"
            payload_data = payload.model_dump()
            payload_data["id"] = story_id
            story = StoryMeta(
                **payload_data,
                created_at=now,
                updated_at=now,
            )
            conn.execute("INSERT INTO stories (id, data) VALUES (?, ?)", (story.id, self._dump(story.model_dump(mode='json'))))
            bible = BibleContent(
                static_facts=normalize_list([f"{story.title}의 기본 전제: {story.premise}"]),
                rules=normalize_list(story.constraints),
                forbidden=normalize_list(story.constraints),
                motifs=normalize_list(story.themes[:3]),
            )
            conn.execute("INSERT OR REPLACE INTO bibles (story_id, data) VALUES (?, ?)", (story.id, self._dump(bible.model_dump(mode='json'))))
            initial_state = StoryState()
            conn.execute(
                "INSERT INTO state_snapshots (story_id, scene_id, scene_index, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (story.id, None, 0, self._dump(initial_state.model_dump(mode='json')), now),
            )
        return story

    def list_stories(self) -> List[StoryMeta]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM stories ORDER BY json_extract(data, '$.updated_at') DESC").fetchall()
        return [StoryMeta(**self._load(row["data"])) for row in rows]

    def get_story(self, story_id: str) -> Optional[StoryMeta]:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM stories WHERE id = ?", (story_id,)).fetchone()
        return StoryMeta(**self._load(row["data"])) if row else None

    def update_story(self, story_id: str, payload: StoryUpdate) -> Optional[StoryMeta]:
        story = self.get_story(story_id)
        if story is None:
            return None
        update_data = {key: value for key, value in payload.model_dump(exclude_none=True).items()}
        update_data["updated_at"] = utcnow_iso()
        updated = story.model_copy(update=update_data)
        with self._connect() as conn:
            conn.execute("UPDATE stories SET data = ? WHERE id = ?", (self._dump(updated.model_dump(mode='json')), story_id))
        return updated

    def get_bible(self, story_id: str) -> BibleContent:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM bibles WHERE story_id = ?", (story_id,)).fetchone()
        return BibleContent(**self._load(row["data"])) if row else BibleContent()

    def save_bible(self, story_id: str, bible: BibleContent) -> BibleContent:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO bibles (story_id, data) VALUES (?, ?)", (story_id, self._dump(bible.model_dump(mode='json'))))
        return bible

    def replace_outline(self, story_id: str, cards: List[OutlineCard]) -> List[OutlineCard]:
        with self._connect() as conn:
            conn.execute("DELETE FROM outline_cards WHERE story_id = ?", (story_id,))
            for card in cards:
                conn.execute(
                    "INSERT INTO outline_cards (id, story_id, scene_index, status, data) VALUES (?, ?, ?, ?, ?)",
                    (card.id, story_id, card.scene_index, card.status, self._dump(card.model_dump(mode='json'))),
                )
        return cards

    def list_outline(self, story_id: str) -> List[OutlineCard]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM outline_cards WHERE story_id = ? ORDER BY scene_index ASC", (story_id,)).fetchall()
        return [OutlineCard(**self._load(row["data"])) for row in rows]

    def mark_outline_used(self, story_id: str, card_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM outline_cards WHERE story_id = ? AND id = ?", (story_id, card_id)).fetchone()
            if row is None:
                return
            card = OutlineCard(**self._load(row["data"]))
            updated = card.model_copy(update={"status": "done"})
            conn.execute(
                "UPDATE outline_cards SET status = ?, data = ? WHERE id = ?",
                ("done", self._dump(updated.model_dump(mode='json')), card_id),
            )

    def get_next_scene_index(self, story_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(scene_index), 0) AS mx FROM scenes WHERE story_id = ?", (story_id,)).fetchone()
        return int(row["mx"]) + 1

    def save_scene(self, scene_row: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scenes (id, story_id, scene_index, data) VALUES (?, ?, ?, ?)",
                (scene_row["id"], scene_row["story_id"], scene_row["scene_index"], self._dump(scene_row)),
            )
            conn.execute("DELETE FROM scene_candidates WHERE scene_id = ?", (scene_row["id"],))
            for candidate in candidates:
                conn.execute(
                    "INSERT INTO scene_candidates (id, scene_id, data) VALUES (?, ?, ?)",
                    (candidate["id"], scene_row["id"], self._dump(candidate)),
                )

    def list_scenes(self, story_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM scenes WHERE story_id = ? ORDER BY scene_index ASC", (story_id,)).fetchall()
        return [self._load(row["data"]) for row in rows]

    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM scenes WHERE id = ?", (scene_id,)).fetchone()
            if row is None:
                return None
            scene = self._load(row["data"])
            candidates = conn.execute("SELECT data FROM scene_candidates WHERE scene_id = ? ORDER BY json_extract(data, '$.candidate_index') ASC", (scene_id,)).fetchall()
        scene["candidates"] = [self._load(item["data"]) for item in candidates]
        return scene

    def save_state_snapshot(self, story_id: str, scene_id: str | None, scene_index: int, state: StoryState) -> Dict[str, Any]:
        payload = state.model_dump(mode='json')
        created_at = utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO state_snapshots (story_id, scene_id, scene_index, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, scene_id, scene_index, self._dump(payload), created_at),
            )
        return {"story_id": story_id, "scene_id": scene_id, "scene_index": scene_index, **payload, "created_at": created_at}

    def get_latest_state(self, story_id: str) -> StoryState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM state_snapshots WHERE story_id = ? ORDER BY scene_index DESC, id DESC LIMIT 1",
                (story_id,),
            ).fetchone()
        return StoryState(**self._load(row["data"])) if row else StoryState()

    def list_state_snapshots(self, story_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT scene_id, scene_index, data, created_at FROM state_snapshots WHERE story_id = ? ORDER BY scene_index ASC, id ASC",
                (story_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            data = self._load(row["data"])
            output.append({
                "story_id": story_id,
                "scene_id": row["scene_id"],
                "scene_index": row["scene_index"],
                **data,
                "created_at": row["created_at"],
            })
        return output

    def save_kg_edges(self, story_id: str, scene_id: str, edges: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kg_edges WHERE story_id = ? AND scene_id = ?", (story_id, scene_id))
            for edge in edges:
                conn.execute(
                    "INSERT INTO kg_edges (story_id, scene_id, data) VALUES (?, ?, ?)",
                    (story_id, scene_id, self._dump(edge)),
                )

    def list_kg_edges(self, story_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM kg_edges WHERE story_id = ? ORDER BY id ASC", (story_id,)).fetchall()
        return [self._load(row["data"]) for row in rows]

    def add_dataset_record(self, story_id: str, scene_id: str | None, pool_type: PoolType, payload: Dict[str, Any]) -> Dict[str, Any]:
        created_at = utcnow_iso()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO dataset_records (story_id, scene_id, pool_type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (story_id, scene_id, pool_type.value, self._dump(payload), created_at),
            )
            record_id = cur.lastrowid
        return {
            "id": record_id,
            "story_id": story_id,
            "scene_id": scene_id,
            "pool_type": pool_type.value,
            "payload": payload,
            "created_at": created_at,
        }

    def list_dataset_records(self, story_id: str, pool_type: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT id, scene_id, pool_type, payload, created_at FROM dataset_records WHERE story_id = ?"
        params: List[Any] = [story_id]
        if pool_type:
            query += " AND pool_type = ?"
            params.append(pool_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        output = []
        for row in rows:
            output.append(
                {
                    "id": row["id"],
                    "story_id": story_id,
                    "scene_id": row["scene_id"],
                    "pool_type": row["pool_type"],
                    "payload": self._load(row["payload"]),
                    "created_at": row["created_at"],
                }
            )
        return output

    def dataset_counts(self, story_id: str) -> Dict[str, int]:
        counts = {pool.value: 0 for pool in PoolType}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pool_type, COUNT(*) AS c FROM dataset_records WHERE story_id = ? GROUP BY pool_type",
                (story_id,),
            ).fetchall()
        for row in rows:
            counts[row["pool_type"]] = int(row["c"])
        return counts

    def save_artifact(self, story_id: str, kind: str, path: str, metadata: Dict[str, Any]) -> ArtifactRecord:
        artifact = ArtifactRecord(id=uuid.uuid4().hex, story_id=story_id, kind=kind, path=path, metadata=metadata)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (id, story_id, kind, path, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (artifact.id, story_id, kind, path, self._dump(metadata), artifact.created_at),
            )
        return artifact

    def list_artifacts(self, story_id: str) -> List[ArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, path, metadata, created_at FROM artifacts WHERE story_id = ? ORDER BY created_at DESC",
                (story_id,),
            ).fetchall()
        return [
            ArtifactRecord(
                id=row["id"],
                story_id=story_id,
                kind=row["kind"],
                path=row["path"],
                metadata=self._load(row["metadata"]) or {},
                created_at=row["created_at"],
            )
            for row in rows
        ]

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import unicodedata

import faiss
import numpy as np


@dataclass(frozen=True)
class EventChunk:
    chunk_id: str
    event_uid: str
    title: str | None
    city: str | None
    first_timing_begin: str | None
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: EventChunk
    score: float


class EmbeddingsClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def load_processed_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not events:
        raise ValueError(f"No processed events found in {path}.")
    return events


def split_text(text: str, *, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - chunk_overlap
    return chunks


def build_event_chunks(
    events: list[dict],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[EventChunk]:
    chunks: list[EventChunk] = []
    for event in events:
        event_uid = str(event.get("uid"))
        for index, text in enumerate(
            split_text(event.get("text_for_rag", ""), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                EventChunk(
                    chunk_id=f"{event_uid}:{index}",
                    event_uid=event_uid,
                    title=event.get("title"),
                    city=event.get("city"),
                    first_timing_begin=event.get("first_timing_begin"),
                    text=text,
                )
            )
    if not chunks:
        raise ValueError("No text chunks were created from processed events.")
    return chunks


def embed_chunks(
    chunks: list[EventChunk],
    embeddings: EmbeddingsClient,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embeddings.embed_documents([chunk.text for chunk in batch]))

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Embedding output shape does not match the chunk count.")
    faiss.normalize_L2(matrix)
    return matrix


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Vectors must be a non-empty 2D matrix.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.replace("'", "-").replace(" ", "-").strip().lower()


def save_vector_store(
    *,
    index: faiss.Index,
    chunks: list[EventChunk],
    output_dir: Path,
    embedding_model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "openagenda.faiss"))
    metadata = {
        "embedding_model": embedding_model,
        "index_type": "IndexFlatIP",
        "distance": "cosine_similarity",
        "total_chunks": len(chunks),
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    (output_dir / "openagenda_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vector_store(vector_store_dir: Path) -> tuple[faiss.Index, list[EventChunk], dict]:
    index_path = vector_store_dir / "openagenda.faiss"
    metadata_path = vector_store_dir / "openagenda_metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Vector store not found in {vector_store_dir}. Run scripts/build_faiss_index.py first."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = [EventChunk(**chunk) for chunk in metadata.get("chunks", [])]
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index size does not match metadata chunk count.")
    return index, chunks, metadata


def similarity_search(
    *,
    query: str,
    index: faiss.Index,
    chunks: list[EventChunk],
    embeddings: EmbeddingsClient,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
) -> list[SearchResult]:
    query_vector = np.asarray([embeddings.embed_query(query)], dtype="float32")
    faiss.normalize_L2(query_vector)

    if allowed_cities:
        normalized_cities = {normalize_text(city) for city in allowed_cities if city}
        positions_to_search = [
            position
            for position, chunk in enumerate(chunks)
            if normalize_text(chunk.city) in normalized_cities
        ]
        if not positions_to_search:
            return []

        vectors = np.asarray(
            [index.reconstruct(int(position)) for position in positions_to_search],
            dtype="float32",
        )
        faiss.normalize_L2(vectors)
        scoped_index = build_faiss_index(vectors)
        scores, scoped_positions = scoped_index.search(query_vector, min(top_k, len(positions_to_search)))

        results: list[SearchResult] = []
        for score, scoped_position in zip(scores[0], scoped_positions[0]):
            if scoped_position < 0:
                continue
            chunk_position = positions_to_search[int(scoped_position)]
            results.append(SearchResult(chunk=chunks[chunk_position], score=float(score)))
        return results

    scores, positions = index.search(query_vector, min(top_k, len(chunks)))

    results: list[SearchResult] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        results.append(SearchResult(chunk=chunks[int(position)], score=float(score)))
    return results

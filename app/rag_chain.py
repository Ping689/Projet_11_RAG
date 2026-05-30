from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.vector_store import (
    EmbeddingsClient,
    SearchResult,
    load_vector_store,
    similarity_search,
    similarity_search_by_vector,
)


DEFAULT_VECTOR_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_store"


class ChatModel(Protocol):
    def invoke(self, input: Any) -> Any:
        ...


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]


@dataclass(frozen=True)
class RagTimings:
    embedding_seconds: float
    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.embedding_seconds + self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True)
class TimedRagAnswer(RagAnswer):
    timings: RagTimings


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
 """          
Tu es le chatbot RAG de Puls-Events. 

Règles:

- Réponds en français, de façon utile et concise. 
- Utilise uniquement les informations présentes dans le contexte.
- Si l'information n'est pas disponible, écris 'Non précisé'.
- N'invente jamais d'information.
- Respecte strictement les dates et périodes demandées.
- Ne recommande pas d'évènement hors période.
- Si le contexte est insuffisant, indique-le clairement.

Pour chaque événement recommandé, utilise le format Markdown suivant :

1. Titre de l'événement

- **Ville :** ...
- **Date :** ...
- **Lieu :** ...
- **Tarif :** ...
- **Public visé :** ...
- **Description :**
...
Courte description de 2 à 3 phrases maximum.
---

Important:
- Laisse une ligne vide entre chaque champ.
- Laisse deux lignes vides entre deux événements.
- N'écris jamais plusieurs informations sur la même ligne.
- N'affiche jamais toutes les informations sur une seule ligne.
- Affiche au maximum {top_k} évènements pertinents.
- La réponse doit être facile à lire dans une interface Streamlit.
- N'utilise pas de tableau.
- N'utilise pas d'emoji.
"""
        ),
        (
            "human",
            "Question utilisateur:\n{question}\n\nNombre d'événements à afficher si possible : {top_k}\n\n"
            "Contexte extrait de la base FAISS:\n{context}",
        ),
    ]
)


def format_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = (
            f"Source {index} | score={result.score:.3f} | "
            f"titre={chunk.title or 'inconnu'} | ville={chunk.city or 'inconnue'} | "
            f"date={chunk.first_timing_begin or 'inconnue'} | uid={chunk.event_uid}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def answer_question(
    *,
    question: str,
    embeddings: EmbeddingsClient,
    chat_model: ChatModel,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
) -> RagAnswer:
    index, chunks, _metadata = load_vector_store(vector_store_dir)
    sources = similarity_search(
        query=question,
        index=index,
        chunks=chunks,
        embeddings=embeddings,
        top_k=top_k,
        allowed_cities=allowed_cities,
    )
    if allowed_cities and not sources:
        cities = ", ".join(sorted(allowed_cities))
        return RagAnswer(
            question=question,
            answer=(
                f"Je n'ai pas trouve de source correspondant au perimetre configure ({cities}) "
                "dans l'index FAISS actuel. Il faut reconstruire la base avec les donnees OpenAgenda "
                "du nouveau perimetre avant de pouvoir repondre correctement."
            ),
            sources=[],
    )
    chain = PROMPT | chat_model | StrOutputParser()
    answer = chain.invoke({"question": question, "top_k": top_k, "context": format_context(sources)})
    return RagAnswer(question=question, answer=answer, sources=sources)


def answer_question_with_timings(
    *,
    question: str,
    embeddings: EmbeddingsClient,
    chat_model: ChatModel,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 5,
    allowed_cities: set[str] | None = None,
) -> TimedRagAnswer:
    index, chunks, _metadata = load_vector_store(vector_store_dir)

    start_time = time.perf_counter()
    query_vector = np.asarray([embeddings.embed_query(question)], dtype="float32")
    embedding_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    sources = similarity_search_by_vector(
        query_vector=query_vector,
        index=index,
        chunks=chunks,
        top_k=top_k,
        allowed_cities=allowed_cities,
    )
    retrieval_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    if allowed_cities and not sources:
        cities = ", ".join(sorted(allowed_cities))
        answer = (
            f"Je n'ai pas trouve de source correspondant au perimetre configure ({cities}) "
            "dans l'index FAISS actuel. Il faut reconstruire la base avec les donnees OpenAgenda "
            "du nouveau perimetre avant de pouvoir repondre correctement."
        )
    else:
        chain = PROMPT | chat_model | StrOutputParser()
        answer = chain.invoke({"question": question, "top_k": top_k, "context": format_context(sources)})
    generation_seconds = time.perf_counter() - start_time

    return TimedRagAnswer(
        question=question,
        answer=answer,
        sources=sources,
        timings=RagTimings(
            embedding_seconds=embedding_seconds,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
        ),
    )

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
            "Tu es le chatbot RAG de Puls-Events. Réponds en français, de façon utile et concise. "
            "Utilise uniquement le contexte fourni. Si le contexte ne permet pas de répondre, dis-le clairement. "
            "Quand tu recommandes des événements, cite le titre, la ville et la date si ces informations sont présentes. "
            "Pour les activités enfants, jeunes ou famille, indique la tranche d'âge uniquement si elle est explicitement "
            "présente dans le contexte. Sinon, écris 'âge non précisé'. Ne transforme pas automatiquement 'jeunes' en "
            "'enfants' si la source ne le dit pas clairement. Respecte strictement les contraintes de date, d'année "
            "et de période demandées par l'utilisateur. Ne recommande pas un événement hors période, même s'il est "
            "sémantiquement proche. Ne suppose jamais qu'un événement est récurrent si le contexte ne l'indique pas "
            "explicitement. Si l'utilisateur demande des recommandations, affiche exactement le nombre d'événements "
            "demandé par le paramètre top_k lorsque suffisamment de sources pertinentes sont disponibles. Si moins "
            "d'événements pertinents sont disponibles, affiche-les tous et explique qu'il n'y en a pas davantage "
            "dans le contexte fourni. Présente chaque événement sous forme de fiche courte, pas comme une simple "
            "liste de titres. Pour chaque fiche, indique le titre, la ville, la date, une description courte et les "
            "informations pratiques disponibles dans le contexte, par exemple le lieu, le tarif, l'inscription, l'accès "
            "ou le public visé. Si plusieurs sources ont le même titre mais des dates différentes, affiche-les comme des "
            "occurrences distinctes. Si la description ou une information pratique n'est pas présente, écris "
            "'non précisé'. N'invente aucune information manquante. N'utilise pas d'emoji ni de pictogramme.",
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

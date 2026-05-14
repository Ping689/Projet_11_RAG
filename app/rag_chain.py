from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.vector_store import EmbeddingsClient, SearchResult, load_vector_store, similarity_search


DEFAULT_VECTOR_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_store"


class ChatModel(Protocol):
    def invoke(self, input: Any) -> Any:
        ...


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es le chatbot RAG de Puls-Events. Reponds en francais, de facon utile et concise. "
            "Utilise uniquement le contexte fourni. Si le contexte ne permet pas de repondre, dis-le clairement. "
            "Quand tu recommandes des evenements, cite le titre, la ville et la date si ces informations sont presentes.",
        ),
        (
            "human",
            "Question utilisateur:\n{question}\n\nContexte extrait de la base FAISS:\n{context}",
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
    answer = chain.invoke({"question": question, "context": format_context(sources)})
    return RagAnswer(question=question, answer=answer, sources=sources)

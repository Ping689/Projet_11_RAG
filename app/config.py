"""Shared configuration loading for the project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None
    openagenda_api_key: str | None
    openagenda_agenda_uid: str | None
    openagenda_region: str
    openagenda_city: str
    openagenda_language: str
    openagenda_allowed_cities: str | None


def get_settings() -> Settings:
    return Settings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        openagenda_api_key=os.getenv("OPENAGENDA_API_KEY"),
        openagenda_agenda_uid=os.getenv("OPENAGENDA_AGENDA_UID"),
        openagenda_region=os.getenv("OPENAGENDA_REGION", "Ile-de-France"),
        openagenda_city=os.getenv("OPENAGENDA_CITY", "Paris"),
        openagenda_language=os.getenv("OPENAGENDA_LANGUAGE", "fr"),
        openagenda_allowed_cities=os.getenv("OPENAGENDA_ALLOWED_CITIES"),
    )

"""Search public OpenAgenda agendas to identify a useful agenda UID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.openagenda_client import OpenAgendaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OpenAgenda agendas.")
    parser.add_argument(
        "--search",
        default="Paris culture",
        help="Text search used against OpenAgenda agendas.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="Maximum number of agendas to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    with OpenAgendaClient(settings.openagenda_api_key or "") as client:
        payload = client.search_agendas(search=args.search, size=args.size)

    agendas = payload.get("agendas", [])
    if not agendas:
        print("No agenda found.")
        return

    for agenda in agendas:
        title = agenda.get("title")
        uid = agenda.get("uid")
        description = agenda.get("description", "")
        print(f"UID: {uid}")
        print(f"Title: {title}")
        if description:
            print(f"Description: {description[:180]}")
        print("-" * 60)


if __name__ == "__main__":
    main()

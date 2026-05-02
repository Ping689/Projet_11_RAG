"""Fetch OpenAgenda events for the selected scope and save raw JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.openagenda_client import OpenAgendaClient


RAW_DIR = ROOT_DIR / "data" / "raw"


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch events from OpenAgenda.")
    parser.add_argument(
        "--agenda-uid",
        default=None,
        help="Agenda UID. Falls back to OPENAGENDA_AGENDA_UID from .env.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region filter, for example Ile-de-France.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="City filter, for example Paris.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="How many days in the past should be included.",
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=365,
        help="How many days in the future should be included.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=300,
        help="Batch size per API call. OpenAgenda max is 300.",
    )
    parser.add_argument(
        "--output",
        default="openagenda_events_raw.json",
        help="Output filename written under data/raw.",
    )
    return parser.parse_args()


def build_initial_params(
    *,
    region: str | None,
    city: str | None,
    language: str,
    start_at: datetime,
    end_at: datetime,
    size: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "detailed": 1,
        "monolingual": language,
        "size": min(size, 300),
        "relative[]": ["passed", "current", "upcoming"],
        "timings[gte]": isoformat_z(start_at),
        "timings[lte]": isoformat_z(end_at),
        "state[]": [2],
    }
    if region:
        params["adminLevel1[]"] = [region]
    if city:
        params["adminLevel4[]"] = [city]
    return params


def fetch_all_events(
    client: OpenAgendaClient,
    agenda_uid: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    after: list[str] | None = None

    while True:
        current_params = dict(params)
        if after:
            current_params["after[]"] = after

        payload = client.list_events(agenda_uid, params=current_params)
        batch = payload.get("events", [])
        events.extend(batch)
        after = payload.get("after")

        if not after:
            break

    return events


def main() -> None:
    args = parse_args()
    settings = get_settings()

    agenda_uid = args.agenda_uid or settings.openagenda_agenda_uid
    if not agenda_uid:
        raise ValueError("An agenda UID is required. Set OPENAGENDA_AGENDA_UID or use --agenda-uid.")

    region = args.region or settings.openagenda_region
    city = args.city or settings.openagenda_city
    now = datetime.now(UTC)
    start_at = now - timedelta(days=args.days_back)
    end_at = now + timedelta(days=args.days_forward)

    params = build_initial_params(
        region=region,
        city=city,
        language=settings.openagenda_language,
        start_at=start_at,
        end_at=end_at,
        size=args.size,
    )

    with OpenAgendaClient(settings.openagenda_api_key or "") as client:
        events = fetch_all_events(client, agenda_uid, params)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / args.output
    payload = {
        "fetched_at": isoformat_z(now),
        "agenda_uid": agenda_uid,
        "filters": {
            "region": region,
            "city": city,
            "timings_gte": isoformat_z(start_at),
            "timings_lte": isoformat_z(end_at),
            "language": settings.openagenda_language,
        },
        "total_events": len(events),
        "events": events,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(events)} events to {output_path}")


if __name__ == "__main__":
    main()

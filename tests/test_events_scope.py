from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "openagenda_events_processed.json"
DEFAULT_ALLOWED_CITIES_BY_AGENDA = {
    "95716291": {
        "bagnolet",
        "bobigny",
        "bondy",
        "le-pre-saint-gervais",
        "les-lilas",
        "montreuil",
        "noisy-le-sec",
        "pantin",
        "romainville",
    }
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.replace("'", "-").replace(" ", "-").strip().lower()


def load_payload() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def load_events() -> list[dict]:
    return load_payload()["events"]


def test_processed_dataset_exists() -> None:
    assert DATASET_PATH.exists(), "Run preprocess_events.py before running tests."


def test_events_match_the_selected_geographic_scope() -> None:
    payload = load_payload()
    events = load_events()
    assert events, "Processed dataset is empty."
    agenda_uid = str(payload.get("agenda_uid", ""))
    allowed_cities = DEFAULT_ALLOWED_CITIES_BY_AGENDA.get(agenda_uid, set())

    for event in events:
        city_normalized = normalize_text(event.get("city"))
        if allowed_cities:
            if not city_normalized:
                continue
            assert city_normalized in allowed_cities, (
                f"Event {event['uid']} is outside the selected city scope: {event.get('city')}"
            )
        else:
            region = normalize_text(event.get("region"))
            assert region == "ile-de-france", (
                f"Event {event['uid']} is outside the selected region scope: {event.get('region')}"
            )


def test_events_timings_are_within_the_selected_time_window() -> None:
    payload = load_payload()
    events = load_events()
    filters = payload["filters"]
    lower_bound = datetime.fromisoformat(filters["timings_gte"].replace("Z", "+00:00"))
    upper_bound = datetime.fromisoformat(filters["timings_lte"].replace("Z", "+00:00"))

    for event in events:
        first_timing_begin = event.get("first_timing_begin")
        assert first_timing_begin, f"Event {event['uid']} has no timing."
        first_dt = datetime.fromisoformat(first_timing_begin.replace("Z", "+00:00"))
        assert lower_bound <= first_dt <= upper_bound, (
            f"Event {event['uid']} is outside the selected time window."
        )

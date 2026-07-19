"""NY Open Data (Socrata) ingestors for Powerball and Mega Millions.

These official state datasets mirror the multi-state draw archives and are the
recommended automated backfill source in the research document.  Network
failures raise ``IngestError`` so callers can fall back to CSV imports.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..schema import Draw

_PB_URL = "https://data.ny.gov/resource/d6yy-54nr.json"   # Powerball
_MM_URL = "https://data.ny.gov/resource/5xaw-6ayf.json"   # Mega Millions
_LIMIT = 50000


class IngestError(RuntimeError):
    pass


def _fetch_json(url: str, timeout: float) -> list[dict]:
    req = urllib.request.Request(
        f"{url}?$limit={_LIMIT}", headers={"User-Agent": "lo-toolkit"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise IngestError(f"fetch failed for {url}: {e}") from e


def fetch_ny_powerball(timeout: float = 30.0) -> list[Draw]:
    """Rows have `draw_date`, `winning_numbers` ("n n n n n pb"), `multiplier`."""
    draws = []
    for row in _fetch_json(_PB_URL, timeout):
        nums = [int(x) for x in row["winning_numbers"].split()]
        draws.append(
            Draw(
                game_id="powerball",
                draw_date=row["draw_date"][:10],
                numbers=tuple(nums[:5]),
                bonus=nums[5],
                multiplier=int(row["multiplier"]) if row.get("multiplier") else None,
                source="data.ny.gov/d6yy-54nr",
            )
        )
    return draws


def fetch_ny_megamillions(timeout: float = 30.0) -> list[Draw]:
    """Rows have `draw_date`, `winning_numbers`, `mega_ball`, `multiplier`."""
    draws = []
    for row in _fetch_json(_MM_URL, timeout):
        nums = [int(x) for x in row["winning_numbers"].split()]
        draws.append(
            Draw(
                game_id="megamillions",
                draw_date=row["draw_date"][:10],
                numbers=tuple(nums[:5]),
                bonus=int(row["mega_ball"]) if row.get("mega_ball") else None,
                multiplier=int(row["multiplier"]) if row.get("multiplier") else None,
                source="data.ny.gov/5xaw-6ayf",
            )
        )
    return draws

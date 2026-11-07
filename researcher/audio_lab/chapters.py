"""Parse a user-pasted tracklist. No network."""

from __future__ import annotations

import re

from audio_lab import config, io

# h:mm:ss or mm:ss. Seconds are always two digits so 1:02:03 is not three pairs.
_TIME = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")


class InvalidTracklist(io.AudioLabError):
    """The pasted text is not a usable rising tracklist."""


def parse_tracklist(text: str) -> list[dict]:
    cues = []
    for line in text.splitlines():
        match = _TIME.search(line)
        if match is None:
            continue
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        if seconds > 59 or (match.group(1) is not None and minutes > 59):
            continue
        start_s = hours * 3600 + minutes * 60 + seconds
        name = " ".join((line[: match.start()] + " " + line[match.end() :]).split())
        name = name.strip("-–—|")
        cues.append({"start_s": float(start_s), "name": name})
    if len(cues) < 3:
        raise InvalidTracklist(
            "invalid_tracklist: kamida 3 ta o'suvchi vaqt belgisi kerak "
            f"(topildi: {len(cues)})."
        )
    if cues[0]["start_s"] > config.SEG_FIRST_CUE_MAX_S:
        raise InvalidTracklist(
            "invalid_tracklist: birinchi belgi 0:00 yoki unga yaqin bo'lishi kerak "
            f"(topildi: {cues[0]['start_s']:.0f} s)."
        )
    for earlier, later in zip(cues, cues[1:]):
        if later["start_s"] <= earlier["start_s"]:
            raise InvalidTracklist(
                "invalid_tracklist: vaqt belgilari qat'iy o'suvchi tartibda emas "
                f"({earlier['start_s']:.0f} s keyin {later['start_s']:.0f} s)."
            )
    return cues

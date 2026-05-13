from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    display_name: str
    department: str | None


def get_current_user() -> CurrentUser:
    """Return the active user.

    Stand-in for the common authentication base described in the design doc.
    A real implementation would resolve the SSO identity here.
    """
    s = get_settings()
    return CurrentUser(
        user_id=s.user_id,
        display_name=s.user_display_name,
        department=s.user_department,
    )

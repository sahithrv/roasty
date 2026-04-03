from __future__ import annotations

from sg_coach.bootstrap import bootstrap
from sg_coach.shared.logging import get_logger


def main() -> int:
    settings = bootstrap()
    logger = get_logger(__name__)
    logger.info(
        "app started",
        extra={
            "persona": settings.default_persona,
            "commentary_enabled": settings.commentary_enabled,
            "grok_model": settings.grok_model,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


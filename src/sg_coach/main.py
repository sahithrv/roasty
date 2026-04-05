from __future__ import annotations

import asyncio

from sg_coach.bootstrap import bootstrap
from sg_coach.orchestrator.gta_wasted_pipeline import run_gta_wasted_pipeline
from sg_coach.shared.logging import get_logger


logger = get_logger(__name__)


def prompt_game_selection() -> str:
    """Ask the user which game pipeline they want to run."""
    print()
    print("Select a game:")
    print("1. GTA")
    print("2. Valorant")
    print("Q. Quit")

    while True:
        choice = input("> ").strip().lower()
        if choice in {"1", "gta"}:
            return "gta"
        if choice in {"2", "valorant"}:
            return "valorant"
        if choice in {"q", "quit", "exit"}:
            return "quit"
        print("Enter 1 for GTA, 2 for Valorant, or Q to quit.")


def wait_for_user_ready(game_name: str) -> bool:
    """Let the user boot the game manually before detectors start."""
    print()
    print(f"Boot {game_name} yourself, then come back here.")
    print("Press Enter when the game is on-screen and ready for detectors.")
    print("Type Q and press Enter to cancel.")

    response = input("> ").strip().lower()
    return response not in {"q", "quit", "exit"}


def run_gta_session(settings) -> int:
    """Run the continuous GTA detector session until the user stops it."""
    print()
    print("Starting GTA detectors.")
    print("They will keep running until you stop the program with Ctrl+C.")
    if settings.realtime_enabled and settings.realtime_enable_user_speech:
        print(f"Hold {settings.realtime_push_to_talk_key!r} to talk to the realtime companion.")
    print()

    try:
        asyncio.run(run_gta_wasted_pipeline(frame_count=None))
    except KeyboardInterrupt:
        print()
        print("GTA detector session stopped by user.")
        logger.info("gta detector session stopped by user")
    return 0


def main() -> int:
    settings = bootstrap()
    logger.info(
        "launcher started",
        extra={
            "persona": settings.default_persona,
            "commentary_enabled": settings.commentary_enabled,
            "grok_model": settings.grok_model,
            "realtime_enabled": settings.realtime_enabled,
        },
    )

    print("Sarcastic Gaming AI Coach")
    print("==========================")

    selection = prompt_game_selection()
    if selection == "quit":
        print("Exiting.")
        return 0

    if selection == "valorant":
        print()
        print("Valorant startup flow is not implemented yet.")
        print("Choose GTA for the live detector path right now.")
        return 0

    if selection == "gta":
        if not wait_for_user_ready("GTA"):
            print("Cancelled before detectors started.")
            return 0
        return run_gta_session(settings)

    # Defensive fallback: the selection helper should already normalize inputs.
    print("No valid game selected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

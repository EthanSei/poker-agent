# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poker Agent is a poker training platform where you play No-Limit Texas Hold'em against configurable AI bots. It tracks every hand, scores your decisions against optimal play (DTO scoring), and shows EV at each decision point so you can identify leaks in your game.

## Tech Stack & Conventions

- **Python 3.12**, async/await throughout
- **pyproject.toml** (hatchling) for dependency management
- **FastAPI** with native WebSocket support for real-time gameplay
- **SQLAlchemy** (async) for ORM — Postgres via asyncpg in production, SQLite via aiosqlite for tests
- **phevaluator** for fast Monte Carlo equity simulations (C++ bindings, ~1.4us/eval)
- **pydantic-settings** for configuration from `.env` (see `.env.example`)
- **ruff** for linting and formatting (line-length 100, target py312)
- **mypy** strict mode for type checking
- **Next.js** (React) for the frontend
- Integer chip amounts everywhere (no floats) — buy-in 10000, BB 100

## Build & Run Commands

```bash
# Setup
make dev                    # Install with dev dependencies
make install                # Install production only

# Development
make test                   # Run all tests
make lint                   # Lint with ruff
make format                 # Format with ruff
make typecheck              # Type check with mypy
make check                  # Lint + typecheck + test

# Run a single test
pytest tests/test_evaluator.py::TestHandRanking::test_flush_beats_straight

# Play via CLI
make play                   # python -m poker.cli

# Run the server
make run                    # uvicorn poker.api.app:app
```

## Architecture

### Dependency rule (strict, one-way only)

```
engine/ --> scoring/ --> bots/ --> api/
```

An import lint test enforces this. Engine must NEVER import from scoring, bots, api, or db.

### Package layout

- **poker/engine/** — Pure synchronous game logic with ZERO framework imports. Cards, hand evaluation, betting state machine, pot/side-pot calculation. `GameState` is the central type. `apply_action(state, action) -> GameState` and `legal_actions(state) -> list[Action]` are the core interface. Mutable dataclasses for in-progress state, frozen for completed records.
- **poker/scoring/** — DTO scoring and EV calculation. Monte Carlo equity via phevaluator. Preflop GTO range lookup tables (in-memory, loaded at startup). Imports engine only.
- **poker/bots/** — Bot AI strategies. `BotStrategy` ABC with `decide(player_view, legal_actions) -> Action`. Strategies: OptimalEV, OptimalEV+Random(noise%), OptimalEV+Bluff(bluff%). Seeded RNG for reproducibility. Imports engine only.
- **poker/api/** — FastAPI app, WebSocket game handler, REST endpoints for hand history/stats, game orchestrator. The orchestrator owns the game loop — it calls engine for state transitions and players for decisions.
- **poker/db/** — SQLAlchemy models (sessions, hands, hand_players, decision_points), async session factory, Alembic migrations.
- **poker/config.py** — pydantic-settings with `.env` loading.

### Key design patterns

- Engine is a pure FSM: `apply_action()` returns new state, `legal_actions()` enumerates valid moves
- Orchestrator pattern: engine never calls players — the API layer dispatches to humans (async WS) or bots (sync)
- Single `GameState` class with `.to_player_view(seat)` method for filtering and `.to_hand_record()` for persistence
- `validate_state()` checks invariants after every action (chip conservation, no negative stacks)
- Only human decision points stored in decision_points table (bot decisions are recomputable)
- Monte Carlo runs via `asyncio.to_thread()` — no ProcessPoolExecutor unless profiling demands it
- DTO scoring runs inline post-hand (no background queue for single-table play)

### Game parameters

- Buy-in: 10,000 chips
- Big blind: 100 (1% of buy-in)
- Small blind: 50
- Max players per table: 10
- Action timeout: 60 seconds (auto-fold)

## Infrastructure

- **Database:** Supabase Postgres. `DATABASE_URL` env var. SQLite for dev/test.
- **Deployment:** GCP Cloud Run (same as arbiter). Dockerfile with multi-stage build.
- **Frontend:** Next.js deployed separately or as static build served by FastAPI.

## Development Workflow

After completing each phase's PRs, run a deep audit to ensure correctness and avoid context creep before moving to the next phase. Compact and verify assumptions haven't drifted.

## Branch Strategy

Each implementation phase is split into focused PRs on feature branches off `main`:
- `phase1/cards-deck` — Card types and deck
- `phase1/hand-evaluator` — Hand evaluation with test fixtures
- `phase1/pot-calculator` — Pot and side-pot math
- `phase1/betting-state-machine` — Core FSM (GameState, apply_action, legal_actions)
- `phase1/bots-cli` — Basic bots + CLI game runner
- `phase2/equity-scoring` — Monte Carlo equity + DTO scoring
- `phase2/persistence` — DB models, migrations, hand history
- `phase2/rest-api` — REST endpoints for history and stats
- `phase3/websocket-ui` — WebSocket handler + React frontend
- `phase4/multiplayer` — Auth, lobby, multi-table, deployment

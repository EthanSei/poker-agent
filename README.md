# poker-agent

Texas Hold'em training platform. Play No-Limit Hold'em against configurable AI bots that use equity-based strategies with tunable aggression, randomness, and bluff frequencies. Every decision is scored against optimal play (DTO scoring) with EV annotations so you can identify and fix leaks in your game.

## Features

- **Custom NLHE engine** built from scratch — hand evaluation, betting state machine, side pots
- **Configurable bot opponents** (up to 9): optimal EV, optimal + random noise, optimal + bluff %
- **DTO scoring**: each decision scored against GTO preflop ranges + Monte Carlo equity
- **Hand history**: full replay with EV at every decision point
- **Session tracking**: stack progression, running DTO score, leak identification
- **Real-time web UI**: play via browser with WebSocket-powered game table
- **Multiplayer-ready**: architecture supports real human opponents (future)

## Status

### Phase 1: Core Engine + CLI
| PR | Branch | Description | Status |
|----|--------|-------------|--------|
| — | `phase1/cards-deck` | Card, Rank, Suit, Deck types | Planned |
| — | `phase1/hand-evaluator` | 7-card hand evaluation + test fixtures | Planned |
| — | `phase1/pot-calculator` | Pot and side-pot calculation | Planned |
| — | `phase1/betting-state-machine` | GameState FSM, apply_action, legal_actions | Planned |
| — | `phase1/bots-cli` | Bot strategies + CLI runner | Planned |

### Phase 2: Scoring + Persistence
| PR | Branch | Description | Status |
|----|--------|-------------|--------|
| — | `phase2/equity-scoring` | Monte Carlo equity + DTO scoring engine | Planned |
| — | `phase2/persistence` | DB models, Alembic migrations, hand history | Planned |
| — | `phase2/rest-api` | REST endpoints for history and stats | Planned |

### Phase 3: Web UI
| PR | Branch | Description | Status |
|----|--------|-------------|--------|
| — | `phase3/websocket-ui` | WebSocket handler + React poker table | Planned |

### Phase 4: Multiplayer + Deploy
| PR | Branch | Description | Status |
|----|--------|-------------|--------|
| — | `phase4/multiplayer` | Auth, lobby, multi-table, Cloud Run deploy | Planned |

## Quick Start

```bash
make dev        # install dependencies
make play       # play via CLI (Phase 1)
make test       # run tests
make run        # start web server (Phase 3+)
```

## Architecture

```
poker/
├── engine/      # Pure game logic (no framework imports)
│   ├── cards.py, evaluator.py, state.py, pot.py, errors.py
├── scoring/     # DTO scoring + Monte Carlo equity
├── bots/        # AI strategies (optimal, random, bluff)
├── api/         # FastAPI + WebSocket + REST
├── db/          # SQLAlchemy models + Alembic
└── config.py    # Settings
```

Strict one-way dependency: `engine → scoring → bots → api`

See [CLAUDE.md](CLAUDE.md) for full architecture details.

## Tech Stack

**Backend**: Python 3.12, FastAPI, SQLAlchemy async, phevaluator, pydantic-settings
**Frontend**: Next.js (React)
**Database**: PostgreSQL (Supabase) / SQLite (dev)
**Deploy**: GCP Cloud Run, Docker

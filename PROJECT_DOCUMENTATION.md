# Meallion Voice AI — Complete Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Use Case](#use-case)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Voice Agent Pipeline](#voice-agent-pipeline)
6. [Agent Tools & Capabilities](#agent-tools--capabilities)
7. [Admin Dashboard](#admin-dashboard)
8. [Database Schema](#database-schema)
9. [External Integrations](#external-integrations)
10. [SIP/Telephony Integration](#siptelephony-integration)
11. [Memory & Knowledge Base System](#memory--knowledge-base-system)
12. [Configuration Management](#configuration-management)
13. [Deployment & Infrastructure](#deployment--infrastructure)
14. [Call Flow Walkthrough](#call-flow-walkthrough)
15. [Security](#security)
16. [Project Structure](#project-structure)

---

## Project Overview

**Meallion Voice AI** is a production-grade AI-powered voice receptionist system built for **Meallion**, a premium Greek food delivery service. The voice agent is named **"Elena"** and handles customer calls in both **English** and **Greek** via web browser and traditional phone lines (SIP/VoIP).

The system provides:
- Real-time voice conversations with customers using AI
- Order tracking and lookup via Shopify integration
- Automated support ticket creation via ClickUp
- A full-featured admin dashboard for managing the agent's behavior, knowledge base, prompts, and call analytics
- Multi-language support (English and Greek with separate agent workers)
- SIP/VoIP phone integration for receiving traditional phone calls

---

## Use Case

### Primary Use Case: AI Voice Receptionist for Food Delivery

Elena serves as an automated customer service representative that:

1. **Answers incoming calls** — Both web-based (WebRTC) and traditional phone calls (SIP)
2. **Handles order inquiries** — Customers can check order status by providing their order number or phone number
3. **Provides information** — Answers FAQs about the brand, menu, delivery policies, and more from a managed knowledge base
4. **Creates support tickets** — When issues can't be resolved on the call, Elena collects customer details and creates a ticket in ClickUp
5. **Speaks naturally** — Uses high-quality text-to-speech (ElevenLabs) with a custom female voice
6. **Understands speech** — Uses Deepgram for accurate speech-to-text transcription
7. **Operates 24/7** — No human staffing required for routine inquiries

### Business Value

- Reduces customer wait times to zero
- Handles routine inquiries without human intervention
- Operates around the clock
- Provides consistent, accurate information from a managed knowledge base
- Escalates complex issues to human agents via support tickets
- Supports bilingual customers (Greek and English)

---

## Technology Stack

### Backend (Python 3.11)

| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.109.2 | REST API server |
| LiveKit Agents SDK | 0.12+ | Voice pipeline framework |
| SQLAlchemy 2.0 (async) | 2.0+ | PostgreSQL ORM |
| asyncpg | 0.29+ | Async PostgreSQL driver |
| Alembic | 1.13+ | Database migrations |
| Pydantic Settings | 2.1+ | Configuration management |
| bcrypt | 4.0+ | Password hashing |
| PyJWT | 2.8+ | JWT authentication |
| httpx | 0.27.2 | Async HTTP client |
| aiosmtplib | 3.0.1 | Async email sending |
| uvicorn | 0.27.1 | ASGI server |

### AI & Voice Services

| Service | Purpose |
|---------|---------|
| OpenAI GPT-4o-mini | LLM for conversation and reasoning |
| ElevenLabs (eleven_multilingual_v2) | Text-to-Speech synthesis |
| Deepgram Nova-3 | Speech-to-Text transcription |
| Silero VAD | Voice Activity Detection |
| OpenAI Whisper (fallback) | Backup STT when Deepgram unavailable |

### Admin Frontend (React/TypeScript)

| Technology | Purpose |
|-----------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite 5 | Build tool and dev server |
| TailwindCSS 3.4 | Utility-first CSS |
| Radix UI | Accessible component primitives |
| TanStack React Query 5 | Server state management |
| Zustand | Client state management |
| Recharts | Analytics charts |
| Axios | HTTP client |
| React Router v6 | Client-side routing |
| Lucide React | Icons |

### Infrastructure

| Technology | Purpose |
|-----------|---------|
| Docker Compose | Multi-container orchestration |
| PostgreSQL 16 | Primary database |
| Redis 7 | LiveKit backing store and caching |
| LiveKit Server v1.8.3 | WebRTC media server |
| LiveKit SIP Bridge | Phone call integration |
| Caddy | Reverse proxy (production) |
| Node.js 20 | Frontend build/dev |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER COMPOSE                                   │
│                                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  FastAPI     │  │  Agent (EN)  │  │  Agent (EL)  │  │ Admin Frontend  │  │
│  │  API Server  │  │  Worker      │  │  Worker      │  │ React (3001)    │  │
│  │  (port 8000) │  │              │  │              │  │                 │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                  │                  │                    │          │
│         │         ┌────────┴──────────────────┴────────┐          │          │
│         │         │        LiveKit Server               │          │          │
│         │         │        (port 7880)                  │          │          │
│         │         │   WebRTC Media + Signaling          │          │          │
│         │         └────────┬───────────────────────────┘          │          │
│         │                  │                                       │          │
│         │         ┌────────┴───────────────────┐                  │          │
│         │         │   LiveKit SIP Bridge       │                  │          │
│         │         │   (host networking)        │                  │          │
│         │         └────────┬───────────────────┘                  │          │
│         │                  │                                       │          │
│  ┌──────┴───────┐  ┌──────┴───────┐                              │          │
│  │ PostgreSQL   │  │    Redis     │                              │          │
│  │ (port 5432)  │  │ (port 6379)  │                              │          │
│  └──────────────┘  └──────────────┘                              │          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
         │                  │                                       │
         │                  │                                       │
    ┌────┴────┐      ┌─────┴─────┐                          ┌─────┴─────┐
    │ Shopify │      │ SIP/VoIP  │                          │  Browser  │
    │  API    │      │ Provider  │                          │  Client   │
    └─────────┘      │ (Yuboto)  │                          └───────────┘
                     └───────────┘
```

### Data Flow

1. **Web Call Flow**: Browser → LiveKit Token (from FastAPI) → LiveKit Room → Agent Worker joins → STT/LLM/TTS pipeline
2. **Phone Call Flow**: SIP Provider → LiveKit SIP Bridge → LiveKit Room → Agent Worker joins → STT/LLM/TTS pipeline
3. **Admin Panel Flow**: React App → FastAPI `/api/admin/*` endpoints → PostgreSQL
4. **Agent Configuration**: Admin Panel → DB update → Agent cache refresh (5-min TTL) → Updated behavior

---

## Voice Agent Pipeline

The voice agent uses LiveKit's `VoicePipelineAgent` which orchestrates a real-time voice conversation loop:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Audio  │────▶│   VAD   │────▶│   STT   │────▶│   LLM   │────▶│   TTS   │
│  Input  │     │ (Silero)│     │(Deepgram)│     │(GPT-4o) │     │(11Labs) │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                       │
                                                       ▼
                                                 ┌───────────┐
                                                 │   Tools   │
                                                 │(Functions)│
                                                 └───────────┘
```

### Pipeline Components

| Stage | Technology | Configuration |
|-------|-----------|---------------|
| Voice Activity Detection | Silero VAD | min_speech_duration=100ms, min_silence_duration=400ms |
| Speech-to-Text | Deepgram Nova-3 | smart_format=True, language=en, model=nova-3 |
| Language Model | OpenAI GPT-4o-mini | Temperature configurable via admin, function calling enabled |
| Text-to-Speech | ElevenLabs | Custom voice ID, stability=0.45, similarity=0.80, speed=0.60 |

### Agent Configuration (Runtime)

All agent parameters are configurable via the admin dashboard without restart:
- LLM model selection
- Voice settings (stability, similarity, speed)
- VAD thresholds
- Silence timeout and prompts
- Order ID digit ranges
- Phone number validation rules

---

## Agent Tools & Capabilities

The agent exposes these tools to the LLM via function calling:

### 1. Order Lookup (`lookup_order`)
- Looks up a Shopify order by order number (3-6 digits)
- Returns brief status: fulfillment status, delivery date, total price
- Uses an in-memory order cache for instant responses

### 2. Order Details (`get_order_details`)
- Fetches full order details including line items
- Can reference the last looked-up order

### 3. Phone Lookup (`lookup_order_by_phone`)
- Finds orders by customer phone number
- Returns the most recent order(s) for that phone

### 4. Knowledge Base Search (`search_knowledge_base`)
- Searches FAQ items by keyword matching
- Falls back to brand information if no match found

### 5. Support Ticket Creation (`create_support_ticket`)
- Collects: customer name, email, issue description
- Creates a task in ClickUp with auto-categorization
- Auto-assigns priority based on issue keywords (refund, delivery, etc.)

### 6. End Session (`end_session`)
- Gracefully terminates the call
- Records call end in database with transcript

### Session State Machine

The agent tracks conversation state to manage multi-turn flows:

```
idle → awaiting_order → checking_order
     → awaiting_phone → checking_phone
     → ticket_name → ticket_email → ticket_issue → ticket_confirm → creating_ticket
```

---

## Admin Dashboard

The admin panel is a React SPA that provides full control over the voice agent:

### Pages

| Page | Functionality |
|------|--------------|
| **Dashboard** | Real-time overview: active calls, today's stats, system health |
| **Knowledge Base** | Manage FAQ items per language, upload JSON, version history with rollback |
| **Prompts** | Edit system prompts, greeting, and closing messages per language (live reload) |
| **Memory** | Long-term Q&A memory pairs — highest priority in agent responses |
| **Calls** | Call history with transcripts, duration, status, sentiment scores |
| **Analytics** | Call volume charts, success rates, hourly breakdown, trends |
| **Sessions** | Live active sessions, ability to terminate or remove participants |
| **SIP Settings** | Configure SIP providers, manage trunks, view connection health |
| **Settings** | Runtime agent settings (LLM model, voice params, VAD config) |
| **Logs** | Audit logs (admin actions) and error logs (system errors) |

### Authentication
- JWT-based authentication with configurable expiry
- bcrypt password hashing
- Token stored in localStorage with auto-refresh
- Audit logging of all admin actions

---

## Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `admin_users` | Admin panel authentication |
| `calls` | Call history (SID, room, caller, type, status, duration, transcript, sentiment) |
| `agent_sessions` | Per-call session data (messages count, tools used, topics, resolution) |
| `agent_memories` | Long-term Q&A memory for agent training |
| `kb_content` | Knowledge base text content per language |
| `kb_items` | Individual FAQ entries (category, question, answer, keywords) |
| `kb_versions` | Knowledge base version history (JSON snapshots) |
| `prompts_content` | System prompts per language |
| `prompts_versions` | Prompt version history |
| `system_settings` | Runtime key-value configuration |
| `languages` | Supported languages registry |
| `call_analytics` | Daily aggregated call statistics |
| `audit_logs` | Admin action audit trail |
| `error_logs` | System error logs |

### SIP Tables

| Table | Purpose |
|-------|---------|
| `sip_providers` | SIP provider configurations (server, credentials, phone numbers) |
| `sip_events` | SIP call events and connection logs |
| `sip_trunk_status` | Trunk health tracking (status, call counts, errors) |
| `sip_config_versions` | SIP configuration version history |

---

## External Integrations

### Shopify (Order Management)

- **API**: Shopify Admin REST API (2024-01)
- **Purpose**: Order lookup by number or phone, status tracking
- **Features**:
  - In-memory order cache with prefetching
  - Subscription order detection
  - Multi-language order formatting (Greek/English)
  - Voice-optimized output (dates, prices spoken naturally)

### ClickUp (Support Tickets)

- **API**: ClickUp REST API v2
- **Purpose**: Create support tickets as tasks
- **Features**:
  - Auto-categorization by issue keywords (refund, delivery, quality, etc.)
  - Priority assignment (urgent/high/normal/low)
  - Markdown-formatted task descriptions
  - Tags for filtering (refund, delivery, ai-escalation, etc.)

### ElevenLabs (Text-to-Speech)

- **Model**: eleven_multilingual_v2 (supports Greek and English)
- **Voice**: Custom female voice (Elena)
- **Settings**: Stability 0.45, Similarity 0.80, Speed 0.60

### Deepgram (Speech-to-Text)

- **Model**: Nova-3
- **Features**: Smart formatting, punctuation, digit recognition
- **Fallback**: OpenAI Whisper if Deepgram unavailable

### OpenAI (Language Model)

- **Model**: GPT-4o-mini
- **Usage**: Conversation management, intent detection, function calling
- **System Prompt**: Dynamically built from DB (memory + KB + instructions)

### SMTP (Email Notifications)

- **Purpose**: Support ticket email notifications
- **Provider**: Configurable (default: Gmail SMTP)

---

## SIP/Telephony Integration

### How Phone Calls Work

1. **SIP Provider** (e.g., Yuboto) routes incoming calls to the LiveKit SIP Bridge
2. **LiveKit SIP Bridge** creates a LiveKit room and connects the audio
3. **Agent Worker** detects the new room and joins automatically
4. **Voice Pipeline** processes audio in real-time (STT → LLM → TTS)
5. **Call ends** when customer hangs up or agent triggers `end_session`

### SIP Configuration

- Providers stored in PostgreSQL (`sip_providers` table)
- Auto-synced to LiveKit on application startup
- Supports multiple providers simultaneously
- IP-based authentication (allowed IPs) or credential-based auth
- E.164 phone number format validation
- Dispatch rules route calls to the correct agent language

### LiveKit SIP Bridge

- Runs with host networking for proper SIP/RTP NAT traversal
- Handles SIP signaling (INVITE, BYE, etc.)
- Bridges SIP audio to WebRTC (LiveKit rooms)

---

## Memory & Knowledge Base System

### Priority System (System Prompt Construction)

The agent's system prompt is built dynamically with this priority order:

```
1. LONG-TERM MEMORY (Highest Priority)
   └── Q&A pairs from admin panel — semantic matching
2. KNOWLEDGE BASE
   └── FAQ content per language from admin panel
3. SYSTEM INSTRUCTIONS
   └── Core behavior, greeting, closing from admin panel
4. FALLBACK
   └── Minimal hardcoded prompt if DB unavailable
```

### Long-Term Memory

- Stored as Q&A pairs in `agent_memories` table
- Injected into system prompt with highest priority
- Semantic/intention-based matching (not exact wording)
- Managed via admin panel (add, edit, activate/deactivate)

### Knowledge Base

- Text content per language stored in `kb_content`
- Individual FAQ items in `kb_items` with categories and keywords
- Version history with rollback capability
- Keyword-based search for tool responses

### Caching Strategy

- 5-minute TTL cache for all DB content
- Background refresh task
- Manual flush available via admin API
- Parallel fetch of KB, prompts, settings, and memory

---

## Configuration Management

### Environment Variables (.env)

All configuration is managed via environment variables using Pydantic Settings:

| Category | Key Variables |
|----------|--------------|
| Server | `HOST`, `PORT`, `DEBUG`, `LOG_LEVEL` |
| LiveKit | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| ElevenLabs | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL` |
| Deepgram | `DEEPGRAM_API_KEY` |
| Shopify | `SHOPIFY_STORE_URL`, `SHOPIFY_ACCESS_TOKEN` |
| ClickUp | `CLICKUP_API_TOKEN`, `CLICKUP_LIST_ID` |
| SIP | `YUBOTO_SIP_SERVER`, `YUBOTO_SIP_USERNAME`, `YUBOTO_SIP_PASSWORD` |
| Database | `POSTGRES_URL`, `DATABASE_URL` |
| Admin | `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_JWT_SECRET` |
| Email | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` |

### Runtime Settings (via Admin Panel)

These settings are stored in PostgreSQL and take effect without restart:
- LLM model and temperature
- Voice settings (stability, similarity, speed)
- VAD thresholds
- Silence timeout and max prompts
- Order ID digit range
- Phone number validation bounds

---

## Deployment & Infrastructure

### Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `api` | Custom (Python 3.11) | 8000 | FastAPI server |
| `agent-en` | Custom (Python 3.11) | — | English voice agent worker |
| `agent-el` | Custom (Python 3.11) | — | Greek voice agent worker |
| `livekit` | livekit/livekit-server:v1.8.3 | 7880, 7881, 7882 | WebRTC media server |
| `sip` | livekit/sip:latest | Host networking | SIP bridge |
| `redis` | redis:7-alpine | 6379 | LiveKit backing store |
| `postgres` | postgres:16-alpine | 5432 | Primary database |
| `frontend` | node:20-alpine | 3000 | Customer-facing web app |
| `admin` | node:20-alpine | 3001 | Admin dashboard |

### Resource Limits

- Agent workers: 0.8 CPU, 1GB RAM each
- LiveKit initialization timeout: 120s (for slow VPS)
- Idle processes: 1 per agent
- Load threshold: 0.9

### Health Checks

- FastAPI: HTTP GET `/health`
- LiveKit: HTTP GET on port 7880
- PostgreSQL: `pg_isready`
- Redis: `redis-cli ping`
- Agents: Process name check in `/proc/1/cmdline`

### Startup Sequence

1. Redis starts → LiveKit starts (depends on Redis)
2. PostgreSQL starts
3. FastAPI starts → initializes DB tables → cleans orphaned calls → syncs LiveKit rooms → syncs SIP providers
4. Agent workers start → connect to LiveKit → begin accepting calls
5. Frontend/Admin start → connect to API

---

## Call Flow Walkthrough

### Web Call (Browser)

```
1. User opens web app in browser
2. Frontend requests LiveKit token from POST /api/token
3. Frontend connects to LiveKit room via WebRTC
4. LiveKit notifies agent worker of new participant
5. Agent worker joins the room
6. Agent speaks greeting: "Hello! I'm Elena from Meallion. How can I help you?"
7. User speaks → Silero VAD detects speech → Deepgram transcribes
8. Transcript sent to GPT-4o-mini with system prompt + tools
9. LLM responds (possibly calling tools like lookup_order)
10. Response text sent to ElevenLabs → audio streamed back to user
11. Loop continues until user says goodbye or silence timeout
12. Agent calls end_session → call recorded in DB with transcript
```

### Phone Call (SIP)

```
1. Customer dials Meallion phone number
2. SIP provider (Yuboto) routes INVITE to LiveKit SIP Bridge
3. SIP Bridge creates LiveKit room, bridges audio
4. Agent worker detects new room, joins
5. Same voice pipeline as web call (steps 6-12 above)
6. When call ends, SIP BYE sent, room destroyed
```

---

## Security

### Authentication & Authorization

- Admin panel uses JWT tokens with configurable expiry (default 24h)
- Passwords hashed with bcrypt
- Token refresh and consecutive 401 error handling
- Audit logging of all admin actions

### API Security

- CORS middleware (configurable origins)
- Bearer token authentication on admin endpoints
- Rate limiting via LiveKit SIP flood protection

### Data Security

- Encrypted SIP passwords in database
- Environment variables for all secrets
- No hardcoded credentials in source code
- PostgreSQL connection with SSL support

---

## Project Structure

```
livekit/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Pydantic Settings configuration
│   ├── agents/
│   │   ├── en/
│   │   │   ├── agent.py           # English voice agent (main logic)
│   │   │   ├── prompts.py         # System prompt builder (DB-driven)
│   │   │   └── tools.py           # Agent tools (order, ticket, KB)
│   │   └── el/
│   │       ├── agent.py           # Greek voice agent
│   │       ├── prompts.py         # Greek prompts
│   │       └── tools.py           # Greek tools
│   ├── api/
│   │   ├── admin.py               # Admin dashboard API endpoints
│   │   └── health.py              # Health check endpoint
│   ├── models/
│   │   ├── base.py                # SQLAlchemy base model
│   │   └── admin.py               # All database models
│   ├── services/
│   │   ├── database.py            # Database service (async SQLAlchemy)
│   │   ├── shopify.py             # Shopify order lookup service
│   │   ├── clickup.py             # ClickUp ticket creation service
│   │   ├── email.py               # SMTP email service
│   │   ├── livekit_sip.py         # LiveKit SIP management
│   │   └── livekit_rooms.py       # LiveKit room management
│   ├── utils/
│   │   ├── greek_numbers.py       # Greek number formatting
│   │   ├── numbers.py             # Digit extraction utilities
│   │   └── voice_formatting.py    # Voice-optimized formatting
│   └── web/
│       └── static/                # Static web frontend files
├── admin/                         # Admin dashboard (React/Vite)
│   ├── src/
│   │   ├── App.tsx                # Main app with routing
│   │   ├── components/            # UI components
│   │   ├── pages/                 # Dashboard pages
│   │   ├── store/                 # Zustand state
│   │   └── lib/                   # API client, utilities
│   └── package.json
├── frontend/                      # Customer-facing web app
├── alembic/                       # Database migrations
├── livekit/                       # LiveKit configuration files
│   ├── livekit.yaml               # LiveKit server config
│   └── sip-config.yaml            # SIP bridge config
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile                     # Python application image
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
└── data/                          # Runtime data (SQLite, room logs)
```

---

## Summary

Meallion Voice AI is a complete, production-ready AI voice receptionist system that combines:

- **Real-time voice AI** (LiveKit + OpenAI + ElevenLabs + Deepgram)
- **Business integrations** (Shopify orders, ClickUp tickets)
- **Full admin control** (React dashboard with live configuration)
- **Phone support** (SIP/VoIP integration)
- **Multi-language** (English and Greek)
- **Containerized deployment** (Docker Compose with health checks)

The system is designed for zero-downtime configuration changes — all prompts, knowledge base content, memory, and agent settings can be updated via the admin panel and take effect within seconds without restarting any services.

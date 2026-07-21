# Teams Bot

Minimal Microsoft 365 Agents SDK bot with RAG Q&A.

## Scripts

- `npm run build`
- `npm run start`
- `npm run dev`
- `npm run typecheck`

## Env vars

Required for `/api/messages`:

- `MicrosoftAppId`
- `MicrosoftAppPassword`
- `MicrosoftAppTenantId`

RAG config:

- `RAG_API_BASE_URL` default: `http://127.0.0.1:8090`
- `RAG_MODEL_NAME` default: `pmt_faq_bot`
- `RAG_TIMEOUT_MS` default: `30000`
- `MAX_TEAMS_MESSAGE_CHARS` default: `12000`

Port:

- `PORT` default: `3978`

## Expected responses

### Health

`GET /healthz`

```json
{
  "status": "ok",
  "service": "teams-bot",
  "version": "0.1.0",
  "node": "v24.x",
  "timestamp": "2026-06-18T...Z"
}
```

### Missing Teams credentials

`POST /api/messages`

```json
{
  "error": "Teams bot is not configured",
  "correlationId": "..."
}
```

## Behavior

- `help` shows commands.
- `status` checks RAG `/healthz`.
- `ask <question>` and `search <query>` call the RAG backend.
- Empty messages return help.
- Natural-language text falls back to RAG.
- Mentions are stripped with simple text cleanup.
- Localhost image markdown is removed/neutralized for Teams.
- Replies are capped to `MAX_TEAMS_MESSAGE_CHARS`.
- Bot typing activity is sent before RAG calls.

## State/privacy decision

- Each Teams message sends only the current query to PMT-FAQ-Bot.
- `teams-bot` stores no chat history in v1.
- Conversation history/state requires an explicit TTL/privacy policy before implementation.
- Future notifications require durable storage, AAD object ID allowlist, rate limiting/backoff, idempotency, and audit logging.

See `docs/state-and-notifications.md`.

## Validation notes

1. Copy `.env.example` to `.env`.
2. `npm install`
3. `npm run build`
4. `npm run start`

## Packaging

- `npm run package:teams` renders `dist/teams-app/manifest.json` and copies icons.
- If `zip` is installed, it also creates `dist/teams-app.zip`.
- Set `TEAMS_APP_ID`, `TEAMS_BOT_DOMAIN`, and either `MICROSOFT_APP_ID` or the runtime `MicrosoftAppId` before packaging.
- `RAG_ASSET_BASE_URL` should be HTTPS-reachable or image links will be neutralized.

See `docs/deployment.md` for deployment/tenant guidance.

## Caveats

- This is compile-only validation unless tested inside a real Teams tenant with Azure Bot/App registration access.
- Teams tenant validation depends on admin consent, app publishing, and a reachable HTTPS endpoint.
- Node v24 builds locally here, but Node LTS is preferred for runtime validation.
- The RAG backend must be reachable at `RAG_API_BASE_URL`.
- Microsoft 365 Agents SDK transitive audit results should be re-checked before production.
- Teams app packaging / manifest is now templated, but tenant deployment still requires Azure Bot/App registration and admin publishing.

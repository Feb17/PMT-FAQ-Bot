# State and notifications decision

## Phase 4 v1

- Each Teams message sends only the current query to PMT-FAQ-Bot.
- `teams-bot` stores no chat history in v1.
- Conversation history/state needs an explicit TTL/privacy policy before implementation.
- Future notifications are deferred.

## Future notification prerequisites

Before notifications are added, design must include:

- durable storage for conversation references
- AAD object ID allowlist
- rate limiting and backoff
- idempotency controls
- audit logging

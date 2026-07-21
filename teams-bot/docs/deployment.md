# Teams bot deployment

## Local compile/smoke validation

1. Copy `.env.example` to `.env`.
2. `npm install`
3. `npm run build`
4. `npm run typecheck`
5. `node smoke-rag-client.mjs`
6. Run the service and verify `GET /healthz`.

## Tenant validation

- Expose the bot with ngrok or similar.
- Configure Azure Bot messaging endpoint to `https://<host>/api/messages`.
- Enable the Teams channel on the Azure Bot resource.
- Use the app package generated from `appPackage/manifest.template.json`.
- Packaging requires `TEAMS_APP_ID`, `TEAMS_BOT_DOMAIN`, and either `MICROSOFT_APP_ID` or the runtime `MicrosoftAppId`.
- Install/test in a real Teams tenant.

## Production requirements

- Azure Bot resource and Entra app registration are required.
- Public HTTPS `/api/messages` endpoint is required.
- PMT-FAQ-Bot RAG backend must be reachable privately or authenticated in production.
- `RAG_ASSET_BASE_URL` must be reachable over HTTPS; otherwise image links are neutralized.
- Corpus ACLs are not enforced by the Teams adapter; confirm corpus is safe for all target users.
- Re-check Agents SDK transitive audit findings before production.
- Org-wide availability is governed by Teams admin app publishing/setup policy.
- Proactive installation/notifications remain deferred and may require Graph/admin consent.

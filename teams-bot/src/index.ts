import dotenv from "dotenv";
import express, { Response } from "express";
import { randomUUID } from "node:crypto";
import packageJson from "../package.json";
import {
  authorizeJWT,
  CloudAdapter,
  Request,
} from "@microsoft/agents-hosting";
import { TeamsActivityHandler } from "@microsoft/agents-hosting-extensions-teams";
import { readConfig, missingTeamsCredentials } from "./config";
import { RagClient } from "./ragClient";
import { MessageHandler } from "./messageHandler";

dotenv.config();
const config = readConfig();
const ragClient = new RagClient(config.ragApiBaseUrl, config.ragModelName, config.ragTimeoutMs);
const messageHandler = new MessageHandler(config, ragClient);

class PmtTeamsBot extends TeamsActivityHandler {
  constructor() {
    super();
    messageHandler.attach(this);
  }
}

const bot = new PmtTeamsBot();
const authConfig = {
  tenantId: config.microsoftAppTenantId,
  clientId: config.microsoftAppId,
  clientSecret: config.microsoftAppPassword,
};
const missingCredentials = missingTeamsCredentials(config);
const adapter = new CloudAdapter();
const app = express();

app.get("/healthz", (_req, res) => {
  res.json({
    status: "ok",
    service: "teams-bot",
    version: packageJson.version,
    node: process.version,
    timestamp: new Date().toISOString(),
  });
});

app.use(express.json());
app.post("/api/messages", async (req: Request, res: Response) => {
  const correlationId = headerValue(req, "x-correlation-id") ?? headerValue(req, "x-ms-client-request-id") ?? cryptoRandomId();
  console.log(`[${correlationId}] /api/messages ${req.method}`);

  if (missingCredentials.length) {
    console.error(`[${correlationId}] missing Teams credentials: ${missingCredentials.join(", ")}`);
    if (!res.headersSent) {
      res.status(500).json({ error: "Teams bot is not configured", correlationId });
    }
    return;
  }

  try {
    await authorizeJWT(authConfig)(req, res, async () => {
      await adapter.process(req, res, async (context) => {
        await bot.run(context);
      });
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[${correlationId}] /api/messages failed`, message);
    if (!res.headersSent) {
      res.status(500).json({ error: "bot processing failed", correlationId });
    }
  }
});

app.listen(config.port, () => {
  console.log(`teams-bot listening on ${config.port}`);
});

function cryptoRandomId(): string {
  return randomUUID();
}

function headerValue(req: Request, name: string): string | undefined {
  const value = req.headers[name.toLowerCase()];
  return Array.isArray(value) ? value[0] : value;
}

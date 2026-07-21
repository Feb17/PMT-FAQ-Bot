export type BotConfig = {
  port: number;
  microsoftAppId?: string;
  microsoftAppPassword?: string;
  microsoftAppTenantId?: string;
  ragApiBaseUrl: string;
  ragModelName: string;
  ragTimeoutMs: number;
  maxTeamsMessageChars: number;
};

const PLACEHOLDERS = new Set(["", "your-app-id", "your-app-password", "your-tenant-id"]);

export function readConfig(): BotConfig {
  const port = parsePort(process.env.PORT ?? "3978");
  return {
    port,
    microsoftAppId: cleanEnv(process.env.MicrosoftAppId),
    microsoftAppPassword: cleanEnv(process.env.MicrosoftAppPassword),
    microsoftAppTenantId: cleanEnv(process.env.MicrosoftAppTenantId),
    ragApiBaseUrl: process.env.RAG_API_BASE_URL?.trim() || "http://127.0.0.1:8090",
    ragModelName: process.env.RAG_MODEL_NAME?.trim() || "pmt_faq_bot",
    ragTimeoutMs: parsePositiveInt(process.env.RAG_TIMEOUT_MS, 30000, "RAG_TIMEOUT_MS"),
    maxTeamsMessageChars: parsePositiveInt(process.env.MAX_TEAMS_MESSAGE_CHARS, 12000, "MAX_TEAMS_MESSAGE_CHARS"),
  };
}

export function missingTeamsCredentials(config: BotConfig): string[] {
  const missing: string[] = [];
  if (!config.microsoftAppId) missing.push("MicrosoftAppId");
  if (!config.microsoftAppPassword) missing.push("MicrosoftAppPassword");
  if (!config.microsoftAppTenantId) missing.push("MicrosoftAppTenantId");
  return missing;
}

function cleanEnv(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed || PLACEHOLDERS.has(trimmed.toLowerCase())) return undefined;
  return trimmed;
}

function parsePort(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) throw new Error(`Invalid PORT: ${value}`);
  return parsed;
}

function parsePositiveInt(value: string | undefined, fallback: number, name: string): number {
  if (!value?.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`Invalid ${name}: ${value}`);
  return parsed;
}

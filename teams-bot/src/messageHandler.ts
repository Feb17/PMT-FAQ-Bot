import { TeamsActivityHandler } from "@microsoft/agents-hosting-extensions-teams";
import { BotConfig } from "./config";
import { RagClient, RagHttpError, RagParseError, RagTimeoutError } from "./ragClient";
import { capTeamsText, sanitizeMarkdownImages, stripMentions } from "./teamsText";

export class MessageHandler {
  constructor(private readonly config: BotConfig, private readonly ragClient: RagClient) {}

  attach(bot: TeamsActivityHandler): void {
    bot.onMessage(async (context, next) => {
      const text = typeof context.activity.text === "string" ? context.activity.text : "";
      await this.handleMessage(text, async (activity) => {
        await context.sendActivity(activity as any);
      });
      await next();
    });
  }

  async handleMessage(rawText: string, send: (activity: string | { type: string }) => Promise<void>): Promise<void> {
    const text = stripMentions(rawText).trim();
    if (!text) return send(helpText());

    const lower = text.toLowerCase();
    if (lower === "help") return send(helpText());
    if (lower === "status") return send(await statusText(this.ragClient));

    const query = lower.startsWith("ask ") || lower.startsWith("search ")
      ? text.replace(/^(ask|search)\s+/i, "").trim()
      : text;
    if (!query) return send("Please include a question after `ask` or `search`.");

    await send({ type: "typing" });
    try {
      const answer = await this.ragClient.chat(query);
      await send(capTeamsText(sanitizeMarkdownImages(answer.text), this.config.maxTeamsMessageChars));
    } catch (error) {
      await send(userFacingRagError(error));
      console.error("RAG request failed", formatRagError(error));
    }
  }
}

function helpText(): string {
  return ["Commands:", "- help", "- status", "- ask <question>", "- search <query>", "Or send a natural-language question."].join("\n");
}

async function statusText(ragClient: RagClient): Promise<string> {
  try { return (await ragClient.healthz()) ? "RAG status: OK" : "RAG status: degraded"; } catch { return "RAG status: unreachable"; }
}

function formatRagError(error: unknown): string {
  if (error instanceof RagTimeoutError) return "RAG request timed out";
  if (error instanceof RagHttpError) return `RAG HTTP ${error.status}`;
  if (error instanceof RagParseError) return "RAG response malformed";
  return error instanceof Error ? error.message : String(error);
}

function userFacingRagError(error: unknown): string {
  if (error instanceof RagTimeoutError) return "The FAQ service took too long to respond. Please try again later.";
  if (error instanceof RagHttpError) return "The FAQ service is currently unavailable. Please try again later.";
  if (error instanceof RagParseError) return "The FAQ service returned an unexpected response. Please try again later.";
  return "I couldn't answer that right now. Please try again later.";
}

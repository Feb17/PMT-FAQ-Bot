export type RagResponse = { text: string };

export class RagTimeoutError extends Error {}
export class RagHttpError extends Error {
  constructor(message: string, public readonly status: number) { super(message); }
}
export class RagParseError extends Error {}

export class RagClient {
  constructor(
    private readonly baseUrl: string,
    private readonly modelName: string,
    private readonly timeoutMs: number,
  ) {}

  async healthz(): Promise<boolean> {
    const { signal, clear } = this.timeoutSignal();
    try {
      const res = await fetch(new URL("/healthz", this.baseUrl), { signal });
      return res.ok;
    } catch (error) {
      throw this.wrapTimeout(error);
    } finally {
      clear();
    }
  }

  async chat(question: string): Promise<RagResponse> {
    const { signal, clear } = this.timeoutSignal();
    let res: Response;
    try {
      res = await fetch(new URL("/v1/chat/completions", this.baseUrl), {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal,
        body: JSON.stringify({ model: this.modelName, stream: false, messages: [{ role: "user", content: question }] }),
      });
      if (!res.ok) throw new RagHttpError(`RAG responded ${res.status}`, res.status);
      const payload = await this.readJson(res);
      const text = payload?.choices?.[0]?.message?.content;
      if (typeof text !== "string" || !text.trim()) throw new RagParseError("Malformed RAG response");
      return { text };
    } catch (error) {
      throw this.wrapTimeout(error);
    } finally {
      clear();
    }
  }

  private timeoutSignal(): { signal: AbortSignal; clear: () => void } {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    timer.unref?.();
    return { signal: controller.signal, clear: () => clearTimeout(timer) };
  }

  private async readJson(res: Response): Promise<any> {
    try {
      return await res.json();
    } catch (error) {
      const wrapped = this.wrapTimeout(error);
      if (wrapped instanceof RagTimeoutError) throw wrapped;
      throw new RagParseError("Malformed RAG response");
    }
  }

  private wrapTimeout(error: unknown): Error {
    if (error instanceof Error && error.name === "AbortError") return new RagTimeoutError("RAG request timed out");
    return error instanceof Error ? error : new Error(String(error));
  }
}

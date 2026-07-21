export function stripMentions(text: string): string {
  return text.replace(/<at>.*?<\/at>/gi, " ").replace(/@\w+/g, " ").replace(/\s+/g, " ").trim();
}

export function sanitizeMarkdownImages(text: string): string {
  return text.replace(/!\[[^\]]*\]\((https?:\/\/(?:127\.0\.0\.1|localhost)[^)]+)\)/gi, "[image omitted]");
}

export function capTeamsText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars - 20)}\n\n[truncated for Teams]`;
}

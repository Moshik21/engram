export const ENTITY_COLORS: Record<string, string> = {
  Person: "#67e8f9",
  Project: "#a78bfa",
  Organization: "#34d399",
  Technology: "#f97316",
  Concept: "#818cf8",
  Event: "#fbbf24",
  Location: "#fb7185",
  Default: "#7a7a94",
};

export function entityColor(type: string): string {
  return ENTITY_COLORS[type] ?? ENTITY_COLORS.Default;
}

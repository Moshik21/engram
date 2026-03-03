/**
 * Activation heatmap color — teal (low) through amber (high).
 * Smoother interpolation with better perceptual uniformity.
 */
export function activationColor(activation: number): string {
  const t = Math.max(0, Math.min(1, activation));
  // Interpolate hue: teal (185°) → amber (30°) through warm arc
  const hue = 185 - t * 155;
  const saturation = 65 + t * 20; // 65% → 85%
  const lightness = 55 + t * 10; // 55% → 65%
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Activation glow color — used for box shadows and glows.
 */
export function activationGlow(activation: number, alpha = 0.3): string {
  const t = Math.max(0, Math.min(1, activation));
  const hue = 185 - t * 155;
  const saturation = 65 + t * 20;
  const lightness = 55 + t * 10;
  return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
}

/**
 * Entity type color palette — carefully chosen for dark backgrounds.
 * Each color is distinct, vibrant, and accessible on dark surfaces.
 */
export const ENTITY_TYPE_COLORS: Record<string, string> = {
  // Lowercase (benchmark corpus)
  person: "#818cf8",      // Indigo
  organization: "#a78bfa", // Violet
  project: "#22d3ee",     // Cyan
  technology: "#34d399",   // Emerald
  concept: "#fbbf24",     // Amber
  location: "#fb7185",    // Rose
  event: "#f472b6",       // Pink

  // Capitalized (existing data)
  Person: "#818cf8",
  Organization: "#a78bfa",
  Project: "#22d3ee",
  Technology: "#34d399",
  Concept: "#fbbf24",
  Location: "#fb7185",
  Event: "#f472b6",
  Other: "#64748b",
};

/**
 * Get entity color with fallback.
 */
export function entityColor(type: string): string {
  return ENTITY_TYPE_COLORS[type] ?? ENTITY_TYPE_COLORS[type.toLowerCase()] ?? "#64748b";
}

/**
 * Get a dimmed version of entity color for backgrounds.
 */
export function entityColorDim(type: string, alpha = 0.12): string {
  const hex = entityColor(type);
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

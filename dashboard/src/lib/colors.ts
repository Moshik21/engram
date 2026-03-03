/**
 * Activation heatmap color — deep blue-violet (resting) → teal → amber-gold (active).
 * Bioluminescent neural aesthetic with warm firing glow.
 */
export function activationColor(activation: number): string {
  const t = Math.max(0, Math.min(1, activation));
  // Interpolate hue: deep blue-violet (240°) → teal (190°) → warm amber-gold (40°)
  let hue: number;
  if (t < 0.4) {
    // Resting → mid: blue-violet to teal
    hue = 240 - (t / 0.4) * 50; // 240° → 190°
  } else {
    // Mid → active: teal to amber-gold
    hue = 190 - ((t - 0.4) / 0.6) * 150; // 190° → 40°
  }
  const saturation = 50 + t * 35; // 50% → 85%
  const lightness = 45 + t * 25; // 45% → 70%
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Activation glow color — used for box shadows and glows.
 */
export function activationGlow(activation: number, alpha = 0.3): string {
  const t = Math.max(0, Math.min(1, activation));
  let hue: number;
  if (t < 0.4) {
    hue = 240 - (t / 0.4) * 50;
  } else {
    hue = 190 - ((t - 0.4) / 0.6) * 150;
  }
  const saturation = 50 + t * 35;
  const lightness = 45 + t * 25;
  return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
}

/**
 * Entity type color palette — bioluminescent neural aesthetic.
 * Each color evokes a different neural region or signaling pathway.
 */
export const ENTITY_TYPE_COLORS: Record<string, string> = {
  // Lowercase (benchmark corpus)
  person: "#c4b5fd",       // Soft violet — cortical neurons
  organization: "#a78bfa", // Deeper violet — association areas
  project: "#67e8f9",      // Bright cyan — active pathways
  technology: "#6ee7b7",   // Seafoam green — growth signals
  concept: "#fcd34d",      // Warm gold — firing neurons
  location: "#fca5a5",     // Soft coral — spatial mapping
  event: "#f9a8d4",        // Pink — temporal markers

  // Capitalized (existing data)
  Person: "#c4b5fd",
  Organization: "#a78bfa",
  Project: "#67e8f9",
  Technology: "#6ee7b7",
  Concept: "#fcd34d",
  Location: "#fca5a5",
  Event: "#f9a8d4",
  Other: "#94a3b8",        // Cool slate — dormant
};

/**
 * Get entity color with fallback.
 */
export function entityColor(type: string): string {
  return ENTITY_TYPE_COLORS[type] ?? ENTITY_TYPE_COLORS[type.toLowerCase()] ?? "#94a3b8";
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

import { describe, it, expect, vi } from "vitest";
import { activationColor, ENTITY_TYPE_COLORS } from "../lib/colors";
import { formatRelativeTime, debounce } from "../lib/utils";

describe("activationColor", () => {
  it("returns deep blue-violet hsl for activation 0", () => {
    const color = activationColor(0);
    // hue=240 (resting blue-violet), sat=50%, light=45%
    expect(color).toBe("hsl(240, 50%, 45%)");
  });

  it("returns warm amber-gold hsl for activation 1", () => {
    const color = activationColor(1);
    // hue=40 (amber-gold), sat=85%, light=70%
    expect(color).toBe("hsl(40, 85%, 70%)");
  });

  it("returns teal-cyan hsl for activation 0.5", () => {
    const color = activationColor(0.5);
    // hue=165 (teal-cyan at mid), sat=67.5%, light=57.5%
    expect(color).toBe("hsl(165, 67.5%, 57.5%)");
  });
});

describe("ENTITY_TYPE_COLORS", () => {
  it("has colors for standard entity types", () => {
    expect(ENTITY_TYPE_COLORS["Person"]).toBeDefined();
    expect(ENTITY_TYPE_COLORS["Organization"]).toBeDefined();
    expect(ENTITY_TYPE_COLORS["Technology"]).toBeDefined();
  });
});

describe("formatRelativeTime", () => {
  it("returns 'never' for null", () => {
    expect(formatRelativeTime(null)).toBe("never");
  });

  it("returns 'just now' for recent timestamps", () => {
    const now = new Date().toISOString();
    expect(formatRelativeTime(now)).toBe("just now");
  });

  it("returns minutes ago for timestamps within an hour", () => {
    const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();
    expect(formatRelativeTime(tenMinutesAgo)).toBe("10m ago");
  });

  it("returns hours ago for timestamps within a day", () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(twoHoursAgo)).toBe("2h ago");
  });

  it("returns days ago for timestamps within a month", () => {
    const fiveDaysAgo = new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(fiveDaysAgo)).toBe("5d ago");
  });

  it("returns months ago for older timestamps", () => {
    const twoMonthsAgo = new Date(Date.now() - 65 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(twoMonthsAgo)).toBe("2mo ago");
  });
});

describe("debounce", () => {
  it("delays execution", async () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const debounced = debounce(fn, 100);

    debounced();
    debounced();
    debounced();

    expect(fn).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledOnce();

    vi.useRealTimers();
  });
});

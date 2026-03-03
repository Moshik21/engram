import { describe, it, expect, vi } from "vitest";
import { activationColor, ENTITY_TYPE_COLORS } from "../lib/colors";
import { formatRelativeTime, debounce } from "../lib/utils";

describe("activationColor", () => {
  it("returns teal hsl for activation 0", () => {
    const color = activationColor(0);
    // hue=185, sat=65%, light=55%
    expect(color).toBe("hsl(185, 65%, 55%)");
  });

  it("returns warm amber hsl for activation 1", () => {
    const color = activationColor(1);
    // hue=30, sat=85%, light=65%
    expect(color).toBe("hsl(30, 85%, 65%)");
  });

  it("returns intermediate hsl for activation 0.5", () => {
    const color = activationColor(0.5);
    // hue=107.5, sat=75%, light=60%
    expect(color).toBe("hsl(107.5, 75%, 60%)");
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

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { getChordColor, getChordTextColor, isNoChord, parseChord } from "./chordColors";

describe("chordColors", () => {
  it("is deterministic for the same label", () => {
    fc.assert(
      fc.property(fc.string(), (label) => {
        expect(getChordColor(label)).toBe(getChordColor(label));
      })
    );
  });

  it("gives different roots different hues (Property 13)", () => {
    const colors = new Set(
      ["C:maj", "D:maj", "E:maj", "F:maj", "G:maj"].map(getChordColor)
    );
    expect(colors.size).toBe(5);
  });

  it("treats N / X / empty as no-chord gray", () => {
    expect(isNoChord("N")).toBe(true);
    expect(isNoChord("X")).toBe(true);
    expect(isNoChord("")).toBe(true);
    expect(getChordColor("N")).toContain("0%");
    expect(getChordTextColor("N")).toBe("#374151");
  });

  it("parses root and quality", () => {
    expect(parseChord("C:maj7")).toEqual({ root: "C", quality: "maj7" });
    expect(parseChord("A#:min")).toEqual({ root: "A#", quality: "min" });
    expect(parseChord("N")).toEqual({ root: "N", quality: "" });
  });
});

import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import fc from "fast-check";
import VisualizationComponent from "./VisualizationComponent";
import type { ChordAnnotation } from "../types";

describe("VisualizationComponent", () => {
  it("positions segments proportional to their timestamps (Property 12)", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), (n) => {
        const duration = n;
        const annotations: ChordAnnotation[] = Array.from({ length: n }, (_, i) => ({
          startTime: i,
          endTime: i + 1,
          chordLabel: i % 2 === 0 ? "C:maj" : "A:min",
        }));
        const { getAllByTestId, unmount } = render(
          <VisualizationComponent annotations={annotations} duration={duration} />
        );
        const segments = getAllByTestId("chord-segment");
        expect(segments).toHaveLength(n);
        segments.forEach((seg, i) => {
          expect(parseFloat(seg.style.left)).toBeCloseTo((i / duration) * 100, 4);
          expect(parseFloat(seg.style.width)).toBeCloseTo((1 / duration) * 100, 4);
        });
        unmount();
      }),
      { numRuns: 25 }
    );
  });

  it("renders different chord types with distinct colors (Property 13)", () => {
    const annotations: ChordAnnotation[] = [
      { startTime: 0, endTime: 1, chordLabel: "C:maj" },
      { startTime: 1, endTime: 2, chordLabel: "D:maj" },
      { startTime: 2, endTime: 3, chordLabel: "E:min" },
    ];
    const { getAllByTestId } = render(
      <VisualizationComponent annotations={annotations} duration={3} />
    );
    const colors = new Set(
      getAllByTestId("chord-segment").map((s) => s.style.backgroundColor)
    );
    expect(colors.size).toBe(3);
  });

  it("covers the whole duration with the last segment reaching the end", () => {
    const annotations: ChordAnnotation[] = [
      { startTime: 0, endTime: 2, chordLabel: "C:maj" },
      { startTime: 2, endTime: 5, chordLabel: "G:7" },
    ];
    const { getAllByTestId } = render(
      <VisualizationComponent annotations={annotations} duration={5} />
    );
    const segments = getAllByTestId("chord-segment");
    expect(segments[0].getAttribute("data-start")).toBe("0");
    expect(segments[segments.length - 1].getAttribute("data-end")).toBe("5");
  });
});

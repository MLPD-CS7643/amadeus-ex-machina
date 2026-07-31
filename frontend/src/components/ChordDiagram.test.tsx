import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import ChordDiagram from "./ChordDiagram";

describe("ChordDiagram", () => {
  it("renders an svg labelled by the chord name", () => {
    const { getByLabelText } = render(
      <ChordDiagram frets={[null, 3, 2, 0, 1, 0]} name="C" />
    );
    expect(getByLabelText("C chord diagram").tagName.toLowerCase()).toBe("svg");
  });

  it("marks muted strings with an x", () => {
    const { container } = render(
      <ChordDiagram frets={[null, 3, 2, 0, 1, 0]} name="C" />
    );
    expect(container.textContent).toContain("×"); // muted low E
  });

  it("shows a position label for shapes played up the neck", () => {
    // E-shape barre up at fret 6 -> outside the open window, so a "6fr" marker appears.
    const { container } = render(
      <ChordDiagram frets={[6, 8, 8, 7, 6, 6]} name="F#" />
    );
    expect(container.textContent).toContain("6fr");
  });

  it("renders a finger dot per fretted string", () => {
    const { container } = render(<ChordDiagram frets={[null, 3, 2, 0, 1, 0]} />);
    // C major frets 3,2,1 -> three dots (open/muted strings get no dot).
    expect(container.querySelectorAll("circle").length).toBeGreaterThanOrEqual(3);
  });
});

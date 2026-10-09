// The page's colors, read from styles.css: every text and background pair must meet WCAG AA (4.5:1), and
// field borders 3:1, in both the light and the dark theme.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { contrast, luminance } from "./contrast";

const css = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

function variables(block: string): Record<string, string> {
  return Object.fromEntries([...block.matchAll(/--([a-z-]+):\s*(#[0-9a-f]{6});/gi)].map((m) => [m[1], m[2]]));
}

const light = variables(css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {"))));
const darkStart = css.indexOf(":root {", css.indexOf("prefers-color-scheme: dark"));
const dark = variables(css.slice(darkStart, css.indexOf("}", darkStart)));

const TEXT_PAIRS = [
  ["text", "bg"], ["text", "surface"], ["muted", "bg"], ["muted", "surface"], ["accent", "surface"], ["accent", "bg"],
  ["on-accent", "accent"], ["covered-text", "covered-bg"], ["not-covered-text", "not-covered-bg"],
  ["error-text", "error-bg"], ["warn-text", "warn-bg"],
];

describe("contrast", () => {
  it("matches WCAG's own examples", () => {
    expect(contrast("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrast("#777777", "#ffffff")).toBeCloseTo(4.48, 2);
    expect(luminance("#ffffff")).toBe(1);
    expect(() => luminance("blue")).toThrow();
  });

  for (const [name, theme] of [["light", light], ["dark", dark]] as const) {
    it(`meets WCAG AA in the ${name} theme`, () => {
      expect(Object.keys(theme).sort()).toEqual(Object.keys(light).sort()); // both themes set every color
      for (const [fg, bg] of TEXT_PAIRS) expect(contrast(theme[fg], theme[bg]), `${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(theme.border, theme.surface), "border on surface").toBeGreaterThanOrEqual(3);
    });
  }
});

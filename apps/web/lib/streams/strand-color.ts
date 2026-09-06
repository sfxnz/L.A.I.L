/**
 * One colour per strand, shared by its band in the Helix chart, its Sequence
 * row and the spine on its card. Four chart hues × two weights give eight
 * distinct strands before the cycle repeats; tokens only, never literals.
 */
export function strandColor(i: number): string {
  return `var(--color-lab-chart-${(i % 4) + 1})`;
}

export function strandOpacity(i: number): number {
  return Math.floor(i / 4) % 2 ? 0.55 : 0.9;
}

export type ContextChunk = {
  kind:
    | "selection"
    | "mention_file"
    | "mention_folder"
    | "mention_search"
    | "open_tab"
    | "note";
  path?: string;
  label: string;
  body: string;
  /** lower = higher priority */
  priority: number;
};

export const PRIORITY = {
  selection: 10,
  mention: 20,
  active_tab: 30,
  open_tab: 40,
  note: 50,
} as const;

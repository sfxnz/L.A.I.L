"use client";

import { useEffect } from "react";

/** Sets document.title for client pages under the shared root layout. */
export function usePageTitle(title: string) {
  useEffect(() => {
    // Avoid cleanup races with React Strict Mode remounts that snap back to the root default.
    document.title = title.includes("L.A.I.L") ? title : `${title} · L.A.I.L`;
  }, [title]);
}

export type ThemePreference = "system" | "light" | "dark";
export const THEME_STORAGE_KEY = "citequest:appearance";

export function themePreference(value: unknown): ThemePreference {
  return value === "light" || value === "dark" ? value : "system";
}

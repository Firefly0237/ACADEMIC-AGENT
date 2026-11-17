import { useEffect, useId, useState } from "react";
import { THEME_STORAGE_KEY, themePreference, type ThemePreference } from "./preferences";
import "./theme.css";

function savedTheme(): ThemePreference {
  try { return themePreference(window.localStorage.getItem(THEME_STORAGE_KEY)); }
  catch { return "system"; }
}

export function ThemeSwitcher() {
  const id = useId();
  const [theme, setTheme] = useState<ThemePreference>(savedTheme);
  useEffect(() => {
    document.documentElement.dataset.cqTheme = theme;
    try { window.localStorage.setItem(THEME_STORAGE_KEY, theme); } catch { /* Session preference remains usable. */ }
  }, [theme]);
  useEffect(() => {
    function changed(event: StorageEvent) {
      if (event.key === THEME_STORAGE_KEY || event.key === null) setTheme(themePreference(event.newValue));
    }
    window.addEventListener("storage", changed);
    return () => window.removeEventListener("storage", changed);
  }, []);
  return <label className="cq-theme-switcher" htmlFor={id}>
    <span>外观</span>
    <select id={id} value={theme} onChange={event => setTheme(themePreference(event.target.value))}>
      <option value="system">跟随系统</option><option value="light">浅色</option><option value="dark">深色</option>
    </select>
  </label>;
}

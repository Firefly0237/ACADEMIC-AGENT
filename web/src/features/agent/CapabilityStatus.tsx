import { useEffect, useState } from "react";
import { errorText } from "../../api";
import { readTaskCapabilities, type ProfileCapabilities } from "./profiles";

export function CapabilityStatus({ kind }: { kind: string }) {
  const [capabilities, setCapabilities] = useState<ProfileCapabilities>();
  const [error, setError] = useState<unknown>();
  useEffect(() => {
    let cancelled = false;
    readTaskCapabilities().then(value => { if (!cancelled) { setCapabilities(value); setError(undefined); } })
      .catch(failure => { if (!cancelled) { setCapabilities(undefined); setError(failure); } });
    return () => { cancelled = true; };
  }, [kind]);
  const profile = capabilities?.profiles.find(item => item.kind === kind);
  return <p className="muted task-capabilities" aria-label="功能是否可用">
    {error ? errorText(error) : !capabilities ? "正在检查此功能是否可用…"
      : !profile ? "此功能暂未开放，请稍后再试。"
      : profile.validation_state === "development" ? "此功能仍在验证中。结果会说明引用依据与尚未解决的问题，请结合原文判断。"
      : "此功能可以使用，提交时会再检查服务状态。"}
  </p>;
}

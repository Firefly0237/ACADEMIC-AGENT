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
  return <p className="muted task-capabilities" aria-label="新任务执行服务状态">
    {error ? errorText(error) : !capabilities ? "正在核对执行服务…"
      : !profile ? "此功能暂未开放新任务；提交前会重新核对执行服务。"
      : profile.validation_state === "development" ? "当前使用开发验证版本，生成结果会标明证据与缺口。"
      : "当前功能已开放，提交前会重新核对执行版本。"}
  </p>;
}

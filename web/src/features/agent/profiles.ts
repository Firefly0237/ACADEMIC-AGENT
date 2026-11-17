import { api } from "../../api";

export type ProfileCapabilities = {
  schema_version: "new-task-profile-policy/1";
  profiles: { kind: string; profile_id: string; validation_state: "development" | "accepted" }[];
  research_review_profile?: { kind: "research"; profile_id: string; validation_state: "development" | "accepted" } | null;
  research_revision_profile?: { kind: "research"; profile_id: string; validation_state: "development" | "accepted" } | null;
  policy_sha256: string;
};

/** Fetch before a new task; never replace a historical Task's profile. */
export async function newTaskProfile(kind: string): Promise<string> {
  const value = await readTaskCapabilities();
  const entries = value.profiles.filter((item) => item.kind === kind);
  if (entries.length !== 1 || !entries[0].profile_id)
    throw new Error("此功能暂未开放新任务，历史结果仍可查看。");
  return entries[0].profile_id;
}

export async function readTaskCapabilities(signal?: AbortSignal): Promise<ProfileCapabilities> {
  const value = await api<ProfileCapabilities>("/api/agent/capabilities", undefined, undefined, signal);
  if (value.schema_version !== "new-task-profile-policy/1" || !Array.isArray(value.profiles))
    throw new Error("执行服务的版本信息不可用，请恢复服务后再创建任务。");
  if (new Set(value.profiles.map(item => item.kind)).size !== value.profiles.length
      || value.profiles.some(item => typeof item.profile_id !== "string" || !item.profile_id
        || !["development", "accepted"].includes(item.validation_state)))
    throw new Error("执行服务的功能清单不完整，请恢复服务后再创建任务。");
  const review = value.research_review_profile;
  if (review != null && (review.kind !== "research" || typeof review.profile_id !== "string" || !review.profile_id
      || !["development", "accepted"].includes(review.validation_state)))
    throw new Error("评审服务的功能清单不完整，请恢复服务后再创建任务。");
  const revision = value.research_revision_profile;
  if (revision != null && (revision.kind !== "research" || typeof revision.profile_id !== "string" || !revision.profile_id
      || !["development", "accepted"].includes(revision.validation_state)))
    throw new Error("修订服务的功能清单不完整，请恢复服务后再创建任务。");
  return value;
}

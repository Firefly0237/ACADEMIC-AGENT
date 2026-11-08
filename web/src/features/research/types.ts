import type { Session } from "../../api";
import type { EvidenceSelection } from "../workstation/types";

export const BRIEF_FIELDS = [
  "topic",
  "purpose",
  "research_questions",
  "literature_scope",
  "data_sources",
  "methods",
  "baselines",
  "metrics",
  "statistical_plan",
  "resources",
  "timeline",
  "risks",
  "stop_conditions",
  "preliminary_work",
  "expected_outputs",
  "innovation_hypotheses",
] as const;
export type BriefField = (typeof BRIEF_FIELDS)[number];
export const FIELD_LABELS: Record<BriefField, string> = {
  topic: "研究方向或题目",
  purpose: "要解决的问题与意义",
  research_questions: "研究问题",
  literature_scope: "文献范围",
  data_sources: "数据与获取条件",
  methods: "拟采用的方法",
  baselines: "比较基线与对照",
  metrics: "评价指标与单位",
  statistical_plan: "重复实验与不确定性",
  resources: "可用资源与经费",
  timeline: "时间安排",
  risks: "风险与可行性",
  stop_conditions: "停止或调整条件",
  preliminary_work: "已有积累与工作",
  expected_outputs: "预期产物",
  innovation_hypotheses: "待验证的创新假设",
};
export type BriefFact = {
  value: string;
  status: "confirmed" | "explicit_unknown";
  message_id: string;
};
export type BriefProposal = {
  id: string;
  field: BriefField;
  candidate: BriefFact;
  kind: "conflict" | "suggestion";
  state: "open" | "accepted" | "discarded" | "superseded";
};
export type BriefDocument = {
  id: string;
  project_id: string;
  revision: number;
  facts: Partial<Record<BriefField, BriefFact>>;
  proposals: BriefProposal[];
  selected_sources: EvidenceSelection[];
  draft_source_ids?: string[];
};
export type BriefSnapshot = {
  brief: BriefDocument;
  brief_sha256: string;
  deleted: boolean;
  readiness: {
    state: "needs_input" | "ready_for_planning";
    missing_fields: BriefField[];
    questions: Array<{ field: BriefField; question: string }>;
    open_conflicts: string[];
    model_calls_required: number;
    school_format_ready: boolean;
  };
};
export type BriefSummary = {
  brief_id: string;
  project_id: string;
  revision: number;
  topic: string | null;
  created_at: string;
  readiness: BriefSnapshot["readiness"];
};
export type PlanningInput = {
  schema_version: string;
  brief_id: string;
  brief_revision: number;
  brief_sha256: string;
  user_confirmed: Record<string, string>;
  explicit_unknowns: Record<string, string>;
  selected_sources: EvidenceSelection[];
  draft_source_ids?: string[];
  research_status: "proposed_not_executed";
  school_format_ready: false;
};
export type BriefSuggestion = { field: BriefField; value: string };
export type DraftCandidate = {
  id: string;
  field: BriefField;
  value: string;
  status: "unconfirmed";
  evidence: Array<{ ref_id: string; quote: string; span_start: number; span_end: number }>;
};
export type DraftExtraction = {
  candidates?: DraftCandidate[];
  missing_fields?: BriefField[];
  incomplete_fields?: BriefField[];
  completion_status?: string;
};
export type BriefHelpRequest = { brief: BriefDocument; fields: BriefField[] };
export type BriefEditorProps = {
  identity: Session;
  projectId: string;
  initial: BriefSnapshot | null;
  onSaved?: (value: BriefSnapshot) => void;
  onResearchHelp?: (
    request: BriefHelpRequest,
  ) => Promise<BriefSuggestion[] | void> | void;
  onResearchStart?: (input: PlanningInput) => Promise<void> | void;
  onResearchExtract?: (request: BriefHelpRequest) => Promise<void> | void;
};

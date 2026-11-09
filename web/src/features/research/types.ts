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
export type MaterialBibliography = {
  file_id: string;
  file_sha256: string;
  representation_id: string;
  title: string;
  authors: string[];
  year: number;
  doi?: string;
  source_url?: string;
  confirmed: true;
  provenance: "user_confirmed";
  confirmation_message_id?: string;
};
export type BriefDocument = {
  id: string;
  project_id: string;
  revision: number;
  facts: Partial<Record<BriefField, BriefFact>>;
  proposals: BriefProposal[];
  selected_sources: EvidenceSelection[];
  draft_source_ids?: string[];
  material_bibliography?: Record<string, MaterialBibliography> | null;
  school_profile_id?: string | null;
};
export type SchoolProfileOption = {
  id: string;
  label: string;
  source_url: string;
  template_source_url: string | null;
  template_sha256: string;
  section_titles: string[];
  literature_minimum_han: number;
  literature_requirement_source_url: string | null;
  demonstration_minimum_han: number;
  demonstration_maximum_han: number;
};
export type SchoolProfiles = { default_id: string; profiles: SchoolProfileOption[] };
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
    missing_material_bibliography?: string[];
    experiment_readiness?: {
      state: "needs_input" | "ready_for_experiment";
      missing_fields: BriefField[];
      questions: Array<{ field: BriefField; question: string }>;
      open_conflicts: string[];
      model_calls_required: 0;
    };
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
  school_profile_id?: string;
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
export type ResearchStartAction = "plan" | "experiment";
export type BriefEditorProps = {
  identity: Session;
  projectId: string;
  initial: BriefSnapshot | null;
  onSaved?: (value: BriefSnapshot) => void;
  onResearchHelp?: (
    request: BriefHelpRequest,
  ) => Promise<BriefSuggestion[] | void> | void;
  onResearchStart?: (input: PlanningInput, action?: ResearchStartAction) => Promise<void> | void;
  onResearchExtract?: (request: BriefHelpRequest) => Promise<void> | void;
};

import type { ReactNode } from "react";
import type { Paper, PaperRef, Session } from "../../api";
import type {
  BriefHelpRequest,
  BriefSuggestion,
  PlanningInput,
} from "../research/types";

export type EvidenceSelection = {
  kind: "metadata_projection" | "native_representation" | "material_file";
  id: string;
};
export type LibraryItem = {
  library_item_id: string;
  work_id: string;
  title: string | null;
  authors: string[] | null;
  year: number | null;
  revision: number;
  file_ids: string[];
  collection_ids: string[];
  tags: string[];
};
export type Collection = {
  collection_id: string;
  name: string;
  parent_id: string | null;
  revision: number;
};
export type HistoryItem = {
  history_id: string;
  kind: string;
  target_id: string;
  occurred_at: string;
  display_title?: string | null;
  source_availability?: SourceAvailability;
  search?: {
    query: string;
    catalog: "application_papers" | "standard_metadata";
    mode: "hybrid" | "bm25" | "dense";
    field: "text" | "author" | "doi";
    sort: "relevance" | "year_desc" | "title";
    filters: { year_from: number | null; year_to: number | null };
    original_generation_id: string;
    original_epoch: number;
    requires_source_reselection: boolean;
    repeat_semantics: "new_search_on_current_catalog_not_historical_result_replay";
  };
};
export type SourceAvailability =
  "available" | "unavailable" | "temporarily_unavailable";
type NoteContent = {
  note_id: string;
  body: string;
  revision: number;
};
export type Note = NoteContent &
  (
    | { saved_paper_id: string; library_item_id?: never }
    | { library_item_id: string; saved_paper_id?: never }
  );
export type TaskSummary = {
  task_id: string;
  kind: string;
  state: string;
  project_id: string;
  revision: number;
  created_at: string;
  detail_path: string;
};
export type Page<T> = {
  items: T[];
  next_cursor: string | null;
  partial_scan?: boolean;
  source_check_incomplete?: boolean;
  library_changed_during_read?: boolean;
  source_status_counts?: Record<string, number>;
  source_availability?: SourceAvailability;
};
export type TaskLaunch = {
  selected: EvidenceSelection[];
  kind: "qa" | "summary" | "compare" | "stance";
  catalog: "application_papers" | "standard_metadata";
  resumeTaskId?: string;
};
export type SavedPaperItem = {
  saved_paper_id: string;
  paper: PaperRef;
  revision: number;
  deleted: boolean;
  source_availability: SourceAvailability;
  metadata: Paper | null;
  collection_ids: string[];
  tags: string[];
  note_count: number;
  qa_scope: "metadata_only" | "native_text" | "unavailable";
  fulltext_summary_available: boolean;
};
export type WorkspacePanelProps = {
  identity: Session;
  briefRefresh?: { id: string; revision: number };
  renderTask?: (request: TaskLaunch) => ReactNode;
  resolveSelections?: (items: LibraryItem[]) => Promise<EvidenceSelection[]>;
  onOpenWork?: (workId: string) => void;
  onOpenTask?: (taskId: string) => void;
  onResearchHelp?: (
    request: BriefHelpRequest,
  ) => Promise<BriefSuggestion[] | void> | void;
  onResearchStart?: (input: PlanningInput) => Promise<void> | void;
  onResearchExtract?: (request: BriefHelpRequest) => Promise<void> | void;
};

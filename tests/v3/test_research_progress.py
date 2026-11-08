"""Small, pure boundary controls; no database/provider/runtime fixtures."""
from __future__ import annotations

import copy
import unittest

from citequest.agents_v3.research_progress import progress_from_state, public_research_progress, school_partial_progress


class ResearchProgressTests(unittest.TestCase):
    def sample(self):
        return progress_from_state({
            "outline": {"questions": [{"experiment_ids": ["E1"]}, {"experiment_ids": ["E2", "E3"]}]},
            "literature": [{"private_quote": "PRIVATE_SOURCE"}],
            "methods": [{"raw_experiment": "UNREVIEWED_METHOD"}],
            "unit_reviews": [], "global_review": {}, "revision_round": 1,
            "waiting": {"kind": "research_motivation_insufficient", "gaps": ["PRIVATE_GAP"]},
            "plan": {"question": "UNREVIEWED_PLAN"},
        }, {"status": "waiting_input"})

    def test_only_observed_counts_and_protocol_states_are_exposed(self):
        value = self.sample()
        self.assertEqual((value["research_questions_planned"], value["literature_reviews_completed"],
            value["experiments_planned"], value["method_proposals_completed"]), (2, 1, 3, 1))
        self.assertEqual(value["status"], "waiting_input")
        self.assertEqual(value["research_status"], "proposed_not_executed")
        self.assertNotIn("PRIVATE", str(value))
        self.assertNotIn("UNREVIEWED", str(value))
        self.assertFalse(value["school_format_ready"])

    def test_unknown_text_or_extra_fields_cannot_become_progress(self):
        value = self.sample()
        for change in ({"raw_plan": "SECRET"}, {"waiting_kind": "model supplied instruction"},
                       {"literature_reviews_completed": True}, {"literature_reviews_completed": 3}):
            with self.subTest(change=change):
                result = {"kind": "research", "research_progress": {**copy.deepcopy(value), **change}}
                self.assertIsNone(public_research_progress(result))

    def test_legacy_other_kind_and_missing_progress_remain_absent(self):
        self.assertIsNone(public_research_progress({"kind": "research", "plan": {"raw": "SECRET"}}))
        self.assertIsNone(public_research_progress({"kind": "qa", "research_progress": self.sample()}))
        self.assertEqual(public_research_progress({"kind": "research", "research_progress": self.sample()}), self.sample())

    def test_school_stop_exposes_only_retained_counts_not_draft_body(self):
        value = school_partial_progress({"kind": "research", "research_progress": self.sample()},
            {"sections": {"review_scope": {"text": "PRIVATE_UNREVIEWED_DRAFT"}}, "reviews": {}})
        self.assertEqual(value["school_sections_written"], 1)
        self.assertEqual(value["school_section_reviews_completed"], 0)
        self.assertEqual(value["waiting_kind"], "school_writing_incomplete")
        self.assertNotIn("PRIVATE_UNREVIEWED_DRAFT", str(value))


if __name__ == "__main__":
    unittest.main()

"""Pure planning resource observations; no runtime or service is started."""
import unittest

from citequest.agents_v3.research_capacity import outline_capacity, planning_operations


class ResearchCapacityTests(unittest.TestCase):
    def task(self):
        return {"profile": {"model_calls": 64, "tool_calls": 64, "input_tokens": 512000, "output_tokens": 96000},
            "model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def test_exact_existing_role_formula_and_school_reserve_are_visible_before_outline(self):
        self.assertEqual(planning_operations(3, 4), (14, 15))
        value = outline_capacity(self.task(), 36)
        self.assertEqual(value["remaining_model_calls_after_this_outline"], 63)
        self.assertIn({"research_questions": 3, "experiments": 4}, value["admissible_model_and_tool_scope_pairs"])
        self.assertNotIn({"research_questions": 8, "experiments": 8}, value["admissible_model_and_tool_scope_pairs"])
        self.assertTrue(value["token_and_context_fit_not_guaranteed_by_scope_pairs"])

    def test_existing_consumption_is_never_reset(self):
        task = self.task()
        task.update(model_calls=30, tool_calls=64, input_tokens=900, output_tokens=700)
        value = outline_capacity(task, 36)
        self.assertEqual(value["remaining_model_calls_after_this_outline"], 33)
        self.assertEqual(value["admissible_model_and_tool_scope_pairs"], [])
        self.assertEqual((task["model_calls"], task["tool_calls"]), (30, 64))


if __name__ == "__main__":
    unittest.main()

"""Tests for deterministic prospect scoring."""

import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import score_prospect


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"annual_revenue": 100, "tech_stack": ["AWS", "Kafka"]}
        self.offering = {
            "min_annual_revenue": 100,
            "required_tech_stack": ["AWS", "Kafka", "dbt", "Snowflake"],
            "description": "Data platform",
            "target_segment": "Enterprise",
        }

    def score(self):
        return score_prospect.invoke({"prospect_profile": self.profile, "offering": self.offering})

    def test_revenue_fit(self):
        for revenue, expected in [(100, 100), (200, 100), (33.6, 34), (0, 0)]:
            with self.subTest(revenue=revenue):
                self.profile["annual_revenue"] = revenue
                self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], expected)

    def test_technology_percentage(self):
        for stack, expected in [([], 0), ([" aws ", "KAFKA"], 50),
                                (["AWS", "Kafka", "dbt"], 75),
                                (["AWS", "Kafka", "dbt", "Snowflake", "extra"], 100)]:
            with self.subTest(stack=stack):
                self.profile["tech_stack"] = stack
                self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], expected)
        self.offering["required_tech_stack"] = ["AWS", "Kafka", "dbt"]
        self.profile["tech_stack"] = ["AWS"]
        self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], 33)

    def test_invalid_revenues_do_not_produce_scores(self):
        for field, record in [("annual_revenue", self.profile), ("min_annual_revenue", self.offering)]:
            for value in [None, -1, "100", True, float("nan"), float("inf")]:
                with self.subTest(field=field, value=value):
                    record[field] = value
                    result = self.score()
                    self.assertIsNone(result["score"])
                    self.assertIn("error", result)
            record.pop(field)
            self.assertIsNone(self.score()["score"])
            record[field] = 100
        self.offering["min_annual_revenue"] = 0
        self.assertIsNone(self.score()["score"])

    def test_invalid_technology_data_does_not_produce_scores(self):
        for field, record in [("tech_stack", self.profile), ("required_tech_stack", self.offering)]:
            for value in [None, "AWS", [None], [" "]]:
                with self.subTest(field=field, value=value):
                    record[field] = value
                    self.assertIsNone(self.score()["score"])
            record.pop(field)
            self.assertIsNone(self.score()["score"])
            record[field] = ["AWS"]
        self.offering["required_tech_stack"] = []
        self.assertIsNone(self.score()["score"])

    def test_repeatable_score_and_grounded_justification(self):
        result = self.score()
        self.assertEqual(result, self.score())
        self.assertEqual(result["score"], 70)
        self.assertEqual(result["rubric_breakdown"], {
            "revenue_fit": 100, "tech_stack_match": 50, "component_max": 100,
        })
        self.assertIn("Present required technologies: AWS, Kafka.", result["justification"])
        self.assertIn("Missing required technologies: dbt, Snowflake.", result["justification"])
        self.assertIn("0.4 × 100 + 0.6 × 50 = 70.0/100", result["justification"])
        self.profile["annual_revenue"] = 50
        self.assertEqual(self.score()["score"], 50)
        self.offering["target_segment"] = "Mid-Market"
        self.assertEqual(self.score()["score"], 50)

    def test_saved_technology_stack_is_used(self):
        self.profile["prospect_id"] = "test-prospect"
        with patch("gtm_agent.data_service.fetch_tech_stack", return_value=["dbt"]):
            result = self.score()
        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 25)
        self.assertIn("Present required technologies: dbt.", result["justification"])


if __name__ == "__main__":
    unittest.main()

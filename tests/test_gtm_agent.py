import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        agent.data_service._PROFILES.clear()
        self.prospect_id = "LEAD-71001"
        self.offering_id = "OFFER-10005"
        self.result = agent.ProspectScore(
            score=70,
            justification="Required technologies assessed.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=70,
            ),
        )
        self.llm = Mock()
        self.llm.invoke.return_value = self.result
        self.scorer = patch.object(agent, "_scoring_llm", self.llm)
        self.scorer.start()
        self.addCleanup(self.scorer.stop)
        self.addCleanup(agent.data_service._PROFILES.clear)

    def score(self, **overrides):
        return agent.score_prospect.invoke({
            "prospect_id": self.prospect_id,
            "offering_id": self.offering_id,
            **overrides,
        })

    def test_scores_from_ids_and_persists_profile_for_standalone_tool(self):
        self.assertEqual(self.score(), self.result.model_dump())
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(profile["found"])
        self.assertEqual(
            agent.data_service.get_profile_from_db(self.prospect_id),
            {"prospect_profile": profile["prospect_profile"]},
        )
        messages = self.llm.invoke.call_args.args[0]
        offering_json, profile_json = messages[1]["content"].removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering_json), agent.data_service.get_offering(self.offering_id))
        self.assertEqual(json.loads(profile_json), profile["prospect_profile"])

    def test_cached_profile_is_used_with_current_tech_stack(self):
        profile = {"prospect_id": self.prospect_id, "name": "Cached name", "tech_stack": ["Stale"]}
        agent.data_service.save_profile_to_db(self.prospect_id, profile)
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Current"]):
            self.assertEqual(self.score(), self.result.model_dump())
        content = self.llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile, {**profile, "tech_stack": ["Current"]})
        self.assertEqual(profile["tech_stack"], ["Stale"])

    def test_missing_records_do_not_score(self):
        for overrides in ({"prospect_id": "missing"}, {"offering_id": "missing"}):
            with self.subTest(overrides=overrides):
                result = self.score(**overrides)
                self.assertFalse(result["found"])
                self.assertIsNone(result["score"])
        self.llm.invoke.assert_not_called()

    def test_incomplete_offerings_keep_validation_error(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field):
                incomplete = {key: value for key, value in offering.items() if key != field}
                with patch.object(agent.data_service, "get_offering", return_value=incomplete):
                    self.assertEqual(self.score(), {
                        "score": None, "error": "Cannot score without a valid offering.",
                    })
        self.llm.invoke.assert_not_called()

    def test_email_sender_defaults_to_runtime_rep(self):
        runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        result = agent.send_prospect_email.func(
            {"name": "Prospect", "email": "prospect@example.com"},
            "Demo", "Can we schedule a demo?", runtime,
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], agent.data_service.get_rep("rep_amills")["email"])
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["body"], "Can we schedule a demo?")


if __name__ == "__main__":
    unittest.main()

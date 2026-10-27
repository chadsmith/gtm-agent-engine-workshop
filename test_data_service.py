import json
import unittest
from unittest.mock import Mock, patch

from gtm_agent import data_service


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        record = data_service.get_prospect_record(self.prospect_id)
        self.original_stack = record["tech_stack"]
        self.addCleanup(record.__setitem__, "tech_stack", self.original_stack)
        profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)

    def test_update_refreshes_cached_profile_and_scoring_input(self):
        with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
            from gtm_agent import gtm_agent as agent

        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", cached["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

        scoring_llm = Mock()
        with patch.object(agent, "_scoring_llm", scoring_llm):
            agent.score_prospect.invoke({
                "prospect_profile": {"prospect_id": self.prospect_id, "tech_stack": []},
                "offering": data_service.get_offering("OFFER-10004"),
            })
        content = scoring_llm.invoke.call_args.args[0][1]["content"]
        profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        self.assertIn("Terraform", profile["tech_stack"])

    def test_update_without_cached_profile_does_not_duplicate_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Terraform"), 1)
        for technology in self.original_stack:
            self.assertIn(technology, result["tech_stack"])

    def test_missing_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("MISSING-PROSPECT", "Terraform")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("MISSING-PROSPECT"))


if __name__ == "__main__":
    unittest.main()

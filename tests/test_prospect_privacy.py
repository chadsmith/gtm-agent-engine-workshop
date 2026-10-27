import importlib
import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")

data_service = agent.data_service


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.pid = "LEAD-12853"
        self.record = data_service.get_prospect_record(self.pid)
        self.score = agent.ProspectScore(
            score=80,
            justification="Good fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=70, segment_fit=80
            ),
        )
        self.llm = Mock()
        self.llm.invoke.return_value = self.score
        scoring = patch.object(agent, "_scoring_llm", self.llm)
        scoring.start()
        self.addCleanup(scoring.stop)
        tracing = tracing_context(enabled=False)
        tracing.__enter__()
        self.addCleanup(tracing.__exit__, None, None, None)

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for record in data_service.PROSPECTS.values():
            for secret in record["billing_qualification"].values():
                self.assertNotIn(secret, serialized)

    def test_contact_allowlist(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(pid=pid):
                with patch.dict(record, {"extra_private_field": "private"}):
                    result = agent.get_prospect.invoke({"prospect_id": pid})
                self.assertEqual(result, {
                    "found": True,
                    "prospect": {
                        "prospect_id": pid,
                        **{k: record[k] for k in ("name", "email", "disqualified")},
                    },
                })
                self.assert_no_billing(result)

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
        self.assertEqual(data_service.get_profile_from_db("missing"), {
            "prospect_profile": None,
        })

    def test_new_profiles_are_safe_before_traced_save(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(pid=pid):
                saved = []
                save = data_service.save_profile_to_db

                def capture_save(prospect_id, profile):
                    self.assert_no_billing(profile)
                    saved.append(deepcopy(profile))
                    return save(prospect_id, profile)

                with patch.object(data_service, "save_profile_to_db", capture_save):
                    result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                expected = {
                    "prospect_id": pid,
                    **{k: v for k, v in record.items() if k != "billing_qualification"},
                }
                self.assertEqual(result, {"prospect_profile": expected, "found": True})
                self.assertEqual(saved, [expected])
                self.assertEqual(data_service.get_profile_from_db(pid), {
                    "prospect_profile": expected,
                })
                self.assert_no_billing(data_service._PROFILES[pid])
                self.assertIn("billing_qualification", record)

    def test_contaminated_cache_is_safe_at_traced_return(self):
        contaminated = {"prospect_id": self.pid, **deepcopy(self.record)}
        data_service._PROFILES[self.pid] = contaminated
        runs = []
        with tracing_context(enabled="local", client=Mock()):
            result = data_service.get_profile_from_db(
                self.pid, langsmith_extra={"on_end": runs.append}
            )
        self.assertEqual(len(runs), 1)
        self.assert_no_billing(runs[0].outputs)
        self.assert_no_billing(result)
        expected = {k: v for k, v in contaminated.items() if k != "billing_qualification"}
        self.assertEqual(result, {"prospect_profile": expected})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.pid}), {
            "prospect_profile": expected, "found": True,
        })

    def test_direct_scoring_sanitizes_prompt_and_uses_saved_tech_stack(self):
        for pid in (self.pid, None):
            with self.subTest(pid=pid):
                profile = deepcopy(self.record)
                if pid is not None:
                    profile["prospect_id"] = pid
                    profile["tech_stack"] = ["unverified"]
                original = deepcopy(profile)
                offering = data_service.get_offering("OFFER-10001")
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": offering,
                })
                self.assertEqual(result, self.score.model_dump())
                messages = self.llm.invoke.call_args.args[0]
                self.assert_no_billing(messages)
                prompted = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
                self.assertEqual(prompted["tech_stack"], self.record["tech_stack"])
                self.assertEqual(prompted["annual_revenue"], self.record["annual_revenue"])
                self.assertEqual(profile, original)

    def test_invalid_offering_does_not_score(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            result = agent.score_prospect.invoke({
                "prospect_profile": self.record, "offering": offering,
            })
            self.assertIsNone(result["score"])
        self.llm.invoke.assert_not_called()

    def test_normal_scoring_and_email_tool_flows(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service._PROFILES.clear()
                if cached:
                    data_service._PROFILES[self.pid] = {
                        "prospect_id": self.pid, **deepcopy(self.record),
                    }
                contact = agent.get_prospect.invoke({"prospect_id": self.pid})
                profile = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
                offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})
                arguments = {
                    "prospect_profile": profile["prospect_profile"],
                    "offering": offering["offering"],
                }
                self.assert_no_billing(arguments)
                score = agent.score_prospect.invoke(arguments)
                email_arguments = {
                    "prospect": contact["prospect"],
                    "subject": "Follow up", "body": "Thanks for your interest.",
                }
                self.assert_no_billing(email_arguments)
                email = agent.send_prospect_email.func(
                    **email_arguments,
                    runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}}),
                )
                self.assertEqual(email["status"], "sent")
                self.assertEqual(email["to"], self.record["email"])
                self.assertEqual(email["from"], data_service.get_rep("rep_amills")["email"])
                self.assertEqual(email["body"], email_arguments["body"])
                self.assert_no_billing([
                    contact, profile, offering, score, email,
                    data_service.get_profile_from_db(self.pid),
                    self.llm.invoke.call_args.args[0],
                ])


if __name__ == "__main__":
    unittest.main()

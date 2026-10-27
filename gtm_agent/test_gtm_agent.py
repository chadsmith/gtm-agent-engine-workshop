import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def send_email(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect=prospect,
            subject="Introduction",
            body="Hello!",
            runtime=SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )

    def test_disqualified_record_blocks_even_with_stale_contact(self):
        prospect_id = next(
            pid for pid, record in gtm_agent.data_service.PROSPECTS.items()
            if record["disqualified"]
        )
        prospect = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
        prospect["disqualified"] = False
        with patch.object(gtm_agent.uuid, "uuid4", side_effect=AssertionError("Email created")):
            result = self.send_email(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": prospect["email"],
        })

    def test_eligible_record_sends_email(self):
        prospect_id = next(
            pid for pid, record in gtm_agent.data_service.PROSPECTS.items()
            if not record["disqualified"]
        )
        prospect = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
        result = self.send_email(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], "rep@example.com")
        self.assertEqual(result["subject"], "Introduction")
        self.assertEqual(result["body"], "Hello!")
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_record_not_marked_disqualified_preserves_sending(self):
        for record in (None, {}, {"disqualified": False}):
            with self.subTest(record=record), patch.object(
                gtm_agent.data_service, "get_prospect_record", return_value=record
            ):
                result = self.send_email({"email": "prospect@example.com"})
                self.assertEqual(result["status"], "sent")

    def test_missing_email_still_fails(self):
        self.assertEqual(self.send_email({}), {
            "status": "failed", "error": "Prospect record has no email address."
        })


if __name__ == "__main__":
    unittest.main()

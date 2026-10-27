import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    import eval as evaluation
    from gtm_agent import gtm_agent as agent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.requests = []
        self.responses = iter([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"name": "Prospect", "email": "prospect@example.com"},
                    "subject": "Follow up",
                    "body": "Hello",
                },
                "id": "email",
            }]),
            AIMessage(content="Email sent."),
        ])

        def generate(model, messages, **kwargs):
            self.requests.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=next(self.responses))])

        self.model = patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=generate)
        self.model.start()
        self.addCleanup(self.model.stop)

    def test_model_receives_only_gtm_prompt_and_tools(self):
        agent.run_agent("Email the prospect", user_id=agent.REP_IDS[0]["rep_id"])

        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        self.assertEqual(len(self.requests), 3)
        for messages, kwargs in self.requests:
            system_messages = [m for m in messages if isinstance(m, SystemMessage)]
            self.assertEqual(len(system_messages), 1)
            self.assertEqual(system_messages[0].content, agent.SYSTEM_PROMPT)
            self.assertEqual(
                {tool["function"]["name"] for tool in kwargs["tools"]},
                expected_tools,
            )

    def test_runtime_metadata_identifies_rep_and_email_sender(self):
        rep = agent.REP_IDS[-1]
        result = agent.run_agent(
            "Email the prospect", user_id=rep["rep_id"],
            environment="test", thread_id="test-thread",
        )

        outputs = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        self.assertEqual(outputs["get_current_rep"], {"rep": rep, "found": True})
        email = outputs["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], rep["email"])
        self.assertEqual(email["from_name"], rep["name"])
        self.assertEqual(email["to"], "prospect@example.com")
        self.assertEqual(result["reply"], "Email sent.")
        self.assertEqual(str(uuid.UUID(result["run_id"])), result["run_id"])

    def test_evaluation_target_preserves_output_and_sender(self):
        rep = agent.REP_IDS[0]
        result = evaluation.evaluation_target({
            "messages": [{"role": "user", "content": "Email the prospect"}],
            "user_id": rep["rep_id"], "thread_id": "eval-thread",
        })

        self.assertEqual(result["output"], "Email sent.")
        self.assertEqual(str(uuid.UUID(result["run_id"])), result["run_id"])
        messages, _ = self.requests[-1]
        email = next(m for m in messages if isinstance(m, ToolMessage) and m.name == "send_prospect_email")
        self.assertEqual(json.loads(email.content)["from"], rep["email"])


if __name__ == "__main__":
    unittest.main()

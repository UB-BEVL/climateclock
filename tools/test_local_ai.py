import unittest
from unittest.mock import Mock, patch
from models.local_ai import ensure_running, MODEL
from models.weather_intelligence import ResearchModelConfig, ask_research_model

class ManagedLocalTests(unittest.TestCase):
    def test_remote_autostart_rejected(self):
        with self.assertRaises(ValueError): ensure_running("https://example.com/v1",MODEL)
    def test_existing_service_reused(self):
        response=Mock(ok=True)
        response.json.return_value={"data":[{"id":MODEL}]}
        with patch("models.local_ai.requests.get",return_value=response), patch("models.local_ai.subprocess.Popen") as launch:
            ensure_running("http://127.0.0.1:11434/v1",MODEL)
            launch.assert_not_called()
    def test_explicit_managed_request_starts_service(self):
        response=Mock(status_code=200)
        response.json.return_value={"choices":[{"message":{"content":"Verified answer"}}]}
        with patch("models.local_ai.ensure_running") as ensure, patch("models.weather_intelligence.requests.post",return_value=response) as post:
            result=ask_research_model(ResearchModelConfig("http://127.0.0.1:11434/v1","local",MODEL,local_managed=True),[],"test")
            ensure.assert_called_once()
            self.assertEqual(result,"Verified answer")
            self.assertEqual(post.call_args.kwargs["timeout"],180)

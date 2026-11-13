import unittest
from unittest.mock import patch, Mock
import requests
from models.weather_intelligence import ResearchModelConfig, ask_research_model

class ConnectionTests(unittest.TestCase):
    def test_base_url_expands_to_chat_endpoint(self):
        response=Mock(status_code=200)
        response.json.return_value={"choices":[{"message":{"content":"OK"}}]}
        with patch("models.weather_intelligence.requests.post",return_value=response) as post:
            answer=ask_research_model(ResearchModelConfig("http://127.0.0.1:11434/v1","local","test"),[],"test")
            self.assertEqual(answer,"OK")
            self.assertEqual(post.call_args.args[0],"http://127.0.0.1:11434/v1/chat/completions")
    def test_unavailable_local_service_is_actionable(self):
        with patch("models.weather_intelligence.requests.post",side_effect=requests.ConnectionError):
            with self.assertRaisesRegex(RuntimeError,"Local Ollama is not reachable"):
                ask_research_model(ResearchModelConfig("http://127.0.0.1:11434/v1","local","test"),[],"test")

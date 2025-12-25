import os
import unittest
from unittest.mock import MagicMock, patch

from ollama_ocr.client import OllamaClient
from ollama_ocr.ocr_engine import OCREngine
from ollama_ocr.structure_engine import StructureEngine


class TestOllamaOCR(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=OllamaClient)
        # Mock chat response
        self.mock_client.chat.return_value = {"message": {"content": "Mocked Content"}}

    @patch("ollama_ocr.ocr_engine.Image.open")
    def test_ocr_engine_image(self, mock_open):
        engine = OCREngine(self.mock_client)
        # Create a dummy image mock
        mock_img = MagicMock()
        mock_open.return_value = mock_img

        # Create a dummy file
        with open("test_dummy.png", "w") as f:
            f.write("dummy")

        try:
            result = engine.process_file("test_dummy.png")
            self.assertEqual(result, "Mocked Content")
            self.mock_client.chat.assert_called()
        finally:
            if os.path.exists("test_dummy.png"):
                os.remove("test_dummy.png")

    def test_structure_engine(self):
        engine = StructureEngine(self.mock_client)

        # Setup mock to return valid JSON for the structure step
        self.mock_client.chat.return_value = {
            "message": {"content": '{"name": "Test", "value": 123}'}
        }

        spec = {"name": "Extract name"}
        result = engine.extract("Some raw text", spec)

        self.assertEqual(result, {"name": "Test", "value": 123})
        self.mock_client.chat.assert_called()


if __name__ == "__main__":
    unittest.main()

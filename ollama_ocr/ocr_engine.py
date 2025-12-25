import io
from pathlib import Path

from PIL import Image

from .client import OllamaClient

# We use uv to install pdf2image, so we can import it directly.
# However, Poppler must be installed on the system.
try:
    from pdf2image import convert_from_path, pdfinfo_from_path

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class OCREngine:
    def __init__(self, client: OllamaClient, model_name: str = "llama3.2-vision"):
        self.client = client
        self.model_name = model_name

    def process_file(self, file_path: str) -> str:
        """
        Process a file (PDF or Image) and return the combined text content.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        images = []

        if suffix == ".pdf":
            if not PDF_SUPPORT:
                raise ImportError(
                    "pdf2image is not installed. Please install it with `uv add pdf2image` (python dependency) "
                    "AND ensure Poppler is installed on your system (e.g., `brew install poppler` on Mac)."
                )

            # Check for Poppler availability explicitly allows for better error messages
            try:
                # 300 DPI is a good balance for OCR
                images = convert_from_path(str(path), dpi=300)
            except Exception as e:
                if "poppler" in str(e).lower():
                    raise OSError(
                        "Poppler not found. Please install Poppler on your system.\n"
                        "  Mac: brew install poppler\n"
                        "  Ubuntu: sudo apt-get install poppler-utils\n"
                        "  Windows: Download generic binary and add to PATH"
                    ) from e
                raise e

        elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            images = [Image.open(file_path)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        full_text = []
        for i, img in enumerate(images):
            print(f"Processing page/image {i + 1}/{len(images)}...")
            text = self._ocr_image(img)
            full_text.append(text)

        return "\n\n".join(full_text)

    def _ocr_image(self, image: Image.Image) -> str:
        """
        Send image to Ollama vision model for transcription.
        """
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        messages = [
            {
                "role": "user",
                "content": "Transcribe the text in this image exactly. Output only the text content, no commentary.",
                "images": [img_bytes],
            }
        ]

        # Use the chat endpoint
        response = self.client.chat(model=self.model_name, messages=messages)

        return response["message"]["content"]

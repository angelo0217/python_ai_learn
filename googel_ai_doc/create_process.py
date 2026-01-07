"""
用途描述:
此檔案用於建立 Google Cloud Document AI 的處理器 (Processor)。
它可以設定處理器的顯示名稱、類型與所在區域 (Location)。
"""

import os

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

# TODO(developer): Uncomment these variables before running the sample.
# project_id = 'YOUR_PROJECT_ID'
# location = 'YOUR_PROCESSOR_LOCATION' # Format is 'us' or 'eu'
# processor_display_name = 'YOUR_PROCESSOR_DISPLAY_NAME' # Must be unique per project, e.g.: 'My Processor'
# processor_type = 'YOUR_PROCESSOR_TYPE' # Use fetch_processor_types to get available processor types


def create_processor_sample(
    project_id: str, location: str, processor_display_name: str, processor_type: str
) -> None:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    # 2. 改為使用預設憑證 (Application Default Credentials)
    # 只要您有執行過 `gcloud auth application-default login` 或設定 GOOGLE_APPLICATION_CREDENTIALS 即可
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the location
    # e.g.: projects/project_id/locations/location
    parent = client.common_location_path(project_id, location)

    # Create a processor
    processor = client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            display_name=processor_display_name, type_=processor_type
        ),
    )

    # Print the processor information
    print(f"Processor Name: {processor.name}")
    print(f"Processor Display Name: {processor.display_name}")
    print(f"Processor Type: {processor.type_}")


if __name__ == "__main__":
    # 請根據你的 GCP 環境修改以下參數
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = "us"  # 選項: 'us' 或 'eu'
    DISPLAY_NAME = "My-New-Processor"
    # 常見類型如: 'OCR_PROCESSOR', 'FORM_W2_PROCESSOR', 'INVOICE_PROCESSOR'
    PROCESSOR_TYPE = "OCR_PROCESSOR"

    create_processor_sample(
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_display_name=DISPLAY_NAME,
        processor_type=PROCESSOR_TYPE,
    )

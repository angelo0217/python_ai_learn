# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# [START documentai_process_document]
"""
用途描述:
此檔案為基礎範例，展示如何呼叫 Document AI 處理文件。
主要功能包括：
1. 提取文件實體 (Entities) 並將結果轉換為乾淨的 JSON 格式。
2. 查看處理器目前的 Schema 結構。
"""

import json
import os
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

# TODO(developer): Uncomment these variables before running the sample.
# project_id = "YOUR_PROJECT_ID"
# location = "YOUR_PROCESSOR_LOCATION" # Format is "us" or "eu"
# processor_id = "YOUR_PROCESSOR_ID" # Create processor before running sample
# file_path = "/path/to/local/pdf"
# mime_type = "application/pdf" # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
# field_mask = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.
# processor_version_id = "YOUR_PROCESSOR_VERSION_ID" # Optional. Processor version to use


def process_document_sample(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    field_mask: Optional[str] = None,
    processor_version_id: Optional[str] = None,
) -> None:
    # You must set the `api_endpoint` if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        # The full resource name of the processor version, e.g.:
        # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        # The full resource name of the processor, e.g.:
        # `projects/{project_id}/locations/{location}/processors/{processor_id}`
        name = client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load binary data
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    # For more information: https://cloud.google.com/document-ai/docs/reference/rest/v1/ProcessOptions
    # Optional: Additional configurations for processing.
    # process_options = documentai.ProcessOptions(
    #     # Process only specific pages
    #     individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
    #         pages=[1]
    #     )
    # )

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
        field_mask=field_mask,
        # process_options=process_options,
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    print("The document contains the following text:")
    print(document.text)

    print("\n" + "=" * 30)
    print("1. [原始] 處理器提取的實體 (Raw Entities):")

    print(json.dumps(extract_document_entities(document), ensure_ascii=False, indent=2))
    print("=" * 30)


def extract_document_entities(document: documentai.Document) -> dict:
    """
    遞迴提取 Document Entities，將其轉換為一般 Python Dict。
    自動處理重複的 key (轉為 list) 與巢狀結構。
    """

    def _process_entity_list(entities_list) -> dict:
        result = {}
        for entity in entities_list:
            key = entity.type_

            # 判斷是否為巢狀實體 (有子屬性)
            if entity.properties:
                value = _process_entity_list(entity.properties)
            else:
                # 葉節點，直接取文字與正規化值
                value = (
                    entity.normalized_value.text
                    if entity.normalized_value
                    else entity.mention_text
                )

            # 把 items 加入 result，處理重複 key
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = value
        return result

    return _process_entity_list(document.entities)


def print_processor_schema(project_id: str, location: str, processor_id: str) -> None:
    """
    查看並列印處理器的 Schema 配置 (Entity Types & Properties)。
    需要使用 v1beta3 API。
    """
    from google.cloud import documentai_v1beta3 as documentai_beta

    print(f"\n[Schema Info] 正在讀取 Processor ({processor_id}) 的 Schema...")

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai_beta.DocumentServiceClient(client_options=opts)

    # 建立 Schema Resource Path
    name = client.dataset_schema_path(project_id, location, processor_id)

    try:
        schema = client.get_dataset_schema(name=name)
        print(f"Schema Resource Name: {schema.name}")

        # document_schema 應該包含 display_name 與 description
        doc_schema = schema.document_schema
        if hasattr(doc_schema, "display_name"):
            print(f"Display Name: {doc_schema.display_name}")
        if hasattr(doc_schema, "description"):
            print(f"Description: {doc_schema.description}")

        if not doc_schema.entity_types:
            print("   (目前無定義任何標籤)")
            return

        print(f"--- Entity Types ({len(doc_schema.entity_types)}) ---")
        for et in doc_schema.entity_types:
            base_type = et.base_types[0] if et.base_types else "N/A"
            print(f"• Name: {et.name}")
            print(f"  Base Type: {base_type}")
            if et.description:
                print(f"  Description: {et.description}")

            if et.properties:
                print(f"  Properties ({len(et.properties)}):")
                for prop in et.properties:
                    print(f"    - {prop.name} (Type: {prop.value_type})")
                    if prop.description:
                        print(f"      Description: {prop.description}")
            print("")

    except Exception as e:
        print(f"❌ 無法讀取 Schema: {e}")


# [END documentai_process_document]

if __name__ == "__main__":
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = "us"
    PROCESSOR_ID = os.getenv("PROCESSOR_ID")

    FILE_PATH = "/Volumes/Dean512G/gcp/demo.jpg"
    MIME_TYPE = "image/jpeg"

    # 1. 處理文件
    process_document_sample(
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID,
        file_path=FILE_PATH,
        mime_type=MIME_TYPE,
    )

    # 2. 查看 Schema (User Request)
    print_processor_schema(
        project_id=PROJECT_ID, location=LOCATION, processor_id=PROCESSOR_ID
    )

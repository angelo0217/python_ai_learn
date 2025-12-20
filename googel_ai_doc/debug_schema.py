import os

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us"
PROCESSOR_ID = os.getenv("PROCESSOR_ID")


def get_dataset_client(location: str):
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return documentai.DocumentServiceClient(client_options=opts)


def inspect_schema():
    client = get_dataset_client(LOCATION)
    name = client.dataset_schema_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    print(f"Reading schema for: {name}")
    try:
        schema = client.get_dataset_schema(name=name)
        if not schema.document_schema.entity_types:
            print("No entity types found.")
            return

        print(f"{'Entity Name':<30} | {'Base Type':<15} | {'Properties (Children)'}")
        print("-" * 80)

        for et in schema.document_schema.entity_types:
            base = et.base_types[0] if et.base_types else "N/A"
            props = [p.name for p in et.properties]
            print(f"{et.name:<30} | {base:<15} | {props}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_schema()

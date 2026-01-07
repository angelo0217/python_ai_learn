# Copyright 2024 Google LLC
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

"""
用途描述:
此檔案為 Schema 管理的核心工具，提供完整的程式化介面來新增 (Add)、更新 (Update) 與刪除 (Delete) Document AI 處理器的標籤定義。
支援處理複雜的 Entity Type 與 Property 關係。
"""

import os
from typing import Dict, List

from google.api_core.client_options import ClientOptions

# 注意: Schema 管理功能大部分位於 v1beta3 版本中
from google.cloud import documentai_v1beta3 as documentai

# ==========================================
# 基礎工具函式 (Helpers)
# ==========================================


def get_dataset_client(location: str):
    """建立 v1beta3 的 Client，需指定正確的 Endpoint"""
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return documentai.DocumentServiceClient(client_options=opts)


def _get_schema_resource(client, project_id: str, location: str, processor_id: str):
    """
    獲取目前的 Dataset Schema 以及 Root Entity Type。
    回傳: (schema, root_entity_type)
    """
    name = client.dataset_schema_path(project_id, location, processor_id)
    try:
        print("\n🔄 正在讀取 Schema...")
        schema = client.get_dataset_schema(name=name)

        # 尋找 Root Entity Type
        root_entity_type = None
        for et in schema.document_schema.entity_types:
            if (
                "document" in et.base_types
                or et.name == "custom_extraction_document_type"
            ):
                root_entity_type = et
                break

        if not root_entity_type:
            print("❌ 錯誤: 找不到 Root Entity Type (base_type='document')。")
            return None, None

        return schema, root_entity_type
    except Exception as e:
        print(f"❌ 讀取 Schema 失敗: {e}")
        return None, None


def _commit_schema_update(client, schema):
    """執行 Schema 更新請求"""
    try:
        request = documentai.UpdateDatasetSchemaRequest(dataset_schema=schema)
        client.update_dataset_schema(request=request)
        print("✅ Schema 更新成功！變更已生效。")
    except Exception as e:
        print(f"❌ 更新失敗: {e}")


def list_current_labels(project_id: str, location: str, processor_id: str):
    """列出目前處理器 Dataset Schema 中的所有標籤"""
    client = get_dataset_client(location)
    schema, _ = _get_schema_resource(client, project_id, location, processor_id)

    if schema:
        print(f"\n� 目前處理器 (ID: {processor_id}) 的標籤清單:")
        if not schema.document_schema.entity_types:
            print("   (尚無定義任何標籤)")

        for et in schema.document_schema.entity_types:
            print(
                f"   - [{et.name}] (Base Type: {et.base_types[0] if et.base_types else 'unknown'})"
            )
        return schema
    return None


# ==========================================
# 核心功能: 新增 (Add)
# ==========================================


def add_labels(
    project_id: str, location: str, processor_id: str, new_labels: List[Dict[str, str]]
):
    """
    新增標籤至 Schema。若標籤已存在，則會跳過。

    Args:
        new_labels: [{"name": "...", "type": "...", "parent": "...", "description": "..."}]
    """
    client = get_dataset_client(location)
    schema, root_entity_type = _get_schema_resource(
        client, project_id, location, processor_id
    )
    if not schema or not root_entity_type:
        return

    has_changes = False
    existing_names = {et.name for et in schema.document_schema.entity_types}

    print(f"   📍 Root Entity: {root_entity_type.name}")

    for label_info in new_labels:
        label_name = label_info["name"]
        data_type = label_info.get("type", "string")
        parent_name = label_info.get("parent")
        description = label_info.get("description", "")

        # 1. 決定 Parent
        target_parent = root_entity_type
        if parent_name:
            found_parent = next(
                (
                    et
                    for et in schema.document_schema.entity_types
                    if et.name == parent_name
                ),
                None,
            )
            if found_parent:
                target_parent = found_parent
            else:
                print(
                    f"   ⚠️ 找不到父物件 '{parent_name}'，無法新增 '{label_name}' (跳過)"
                )
                continue

        # 2. 檢查是否已存在 (Property 或 EntityType)
        # 檢查 Property 是否存在於 Parent
        prop_exists = any(p.name == label_name for p in target_parent.properties)
        # 檢查 EntityType 是否存在
        type_exists = label_name in existing_names

        if prop_exists or type_exists:
            print(f"   ⚠️ 標籤 '{label_name}' 已存在，跳過新增。")
            continue

        print(
            f"   ➕ 新增標籤: {label_name} (Type: {data_type}) -> Parent: {target_parent.name}"
        )

        # 3. 判斷基本型別 vs Entity
        is_primitive = data_type in [
            "string",
            "date",
            "money",
            "integer",
            "number",
            "address",
            "boolean",
            "datetime",
        ]

        if is_primitive and parent_name:
            # 純屬性 (Property Only)
            new_property = documentai.DocumentSchema.EntityType.Property(
                name=label_name,
                value_type=data_type,
                occurrence_type=documentai.DocumentSchema.EntityType.Property.OccurrenceType.OPTIONAL_ONCE,
            )
            if hasattr(new_property, "description"):
                new_property.description = description

            target_parent.properties.append(new_property)
            has_changes = True

        else:
            # 建立新的 EntityType
            new_entity_type = documentai.DocumentSchema.EntityType(
                name=label_name,
                base_types=[data_type],
                description=description,
            )
            schema.document_schema.entity_types.append(new_entity_type)
            existing_names.add(label_name)

            # 關聯到 Parent
            new_property = documentai.DocumentSchema.EntityType.Property(
                name=label_name,
                value_type=label_name,  # 指向 EntityType 名稱
                occurrence_type=documentai.DocumentSchema.EntityType.Property.OccurrenceType.OPTIONAL_ONCE,
            )
            target_parent.properties.append(new_property)
            has_changes = True

    if has_changes:
        _commit_schema_update(client, schema)
    else:
        print("無任何新增變更。")


# ==========================================
# 核心功能: 更新 (Update)
# ==========================================


def update_labels(
    project_id: str,
    location: str,
    processor_id: str,
    update_labels: List[Dict[str, str]],
):
    """
    更新現有標籤 (描述、型別) 或移動父層。若標籤不存在，則會跳過。
    """
    client = get_dataset_client(location)
    schema, root_entity_type = _get_schema_resource(
        client, project_id, location, processor_id
    )
    if not schema or not root_entity_type:
        return

    has_changes = False

    for label_info in update_labels:
        label_name = label_info["name"]
        data_type = label_info.get("type", "string")
        parent_name = label_info.get("parent")
        description = label_info.get("description", "")

        # 尋找 EntityType
        existing_et = next(
            (et for et in schema.document_schema.entity_types if et.name == label_name),
            None,
        )

        # 簡單起見，我們先掃描 root 和所有 entity types 找是誰擁有這個 property
        # 注意：一個 property name 理論上在同一層只能出現一次，但在不同 parent 下可能重複？
        # Document AI Schema 通常名為全域唯一 (EntityType Name)，Property Name 則依附於 Parent。
        # 這裡假設 label_name 對應 EntityType Name 或 Property Name。

        found_locations = []  # (parent_entity, property)
        for et in schema.document_schema.entity_types:
            for p in et.properties:
                if p.name == label_name:
                    found_locations.append((et, p))

        if not existing_et and not found_locations:
            print(f"   ⚠️ 找不到標籤 '{label_name}'，無法更新 (跳過)。")
            continue

        print(f"   🔧 檢查更新: {label_name}")

        # 1. 更新 EntityType (如果存在)
        if existing_et:
            if existing_et.description != description:
                print(f"      📝 更新描述: {description}")
                existing_et.description = description
                has_changes = True

            # 檢查 Base Type
            curr_base = existing_et.base_types[0] if existing_et.base_types else ""
            if curr_base != data_type:
                print(f"      ⚙️ 更新 Base Type: {curr_base} -> {data_type}")
                del existing_et.base_types[:]
                existing_et.base_types.append(data_type)
                has_changes = True

        # 2. 更新 Properties (Type & Description)
        for parent, prop in found_locations:
            # 如果是 Primitive Property，value_type 是 data_type
            # 如果是 Entity Reference，value_type 是 label_name (通常)

            # 判斷這是一個 Reference 還是 Primitive Property
            # 若 existing_et 存在，則 prop.value_type 應為 label_name
            # 若 existing_et 不存在，則 prop.value_type 應為 primitive type

            if not existing_et:
                # Primitive Property: Update Type & Desc
                if prop.value_type != data_type:
                    print(
                        f"      ⚙️ 更新屬性型別 ({parent.name}): {prop.value_type} -> {data_type}"
                    )
                    prop.value_type = data_type
                    has_changes = True

                if hasattr(prop, "description") and prop.description != description:
                    prop.description = description
                    has_changes = True

        # 3. 處理 Parent 移動 (Move)
        # 如果指定了新的 parent，且當前的 parent 不是新的 parent
        if parent_name:
            new_parent_et = next(
                (
                    et
                    for et in schema.document_schema.entity_types
                    if et.name == parent_name
                ),
                None,
            )
            if not new_parent_et:
                # 特例：若 parent_name 指向 Root (雖然 Root 也在 entity_types 裡，但通常需要特別找)
                pass

            if new_parent_et:
                # 檢查目前是否已經在 new_parent 下
                is_already_child = any(
                    parent.name == parent_name for parent, _ in found_locations
                )

                if not is_already_child:
                    print(f"      🚚 移動 Parent: -> {parent_name}")
                    # 從舊 Parent 移除
                    for parent, prop in found_locations:
                        print(f"         ✂️ 從舊 Parent ({parent.name}) 移除")
                        parent.properties.remove(prop)

                    # 加入新 Parent
                    # 需區分是 Reference 還是 Primitive
                    if existing_et:
                        # Add Ref
                        new_prop = documentai.DocumentSchema.EntityType.Property(
                            name=label_name,
                            value_type=label_name,
                            occurrence_type=documentai.DocumentSchema.EntityType.Property.OccurrenceType.OPTIONAL_ONCE,
                        )
                    else:
                        # Add Primitive
                        new_prop = documentai.DocumentSchema.EntityType.Property(
                            name=label_name,
                            value_type=data_type,
                            occurrence_type=documentai.DocumentSchema.EntityType.Property.OccurrenceType.OPTIONAL_ONCE,
                        )
                        if hasattr(new_prop, "description"):
                            new_prop.description = description

                    new_parent_et.properties.append(new_prop)
                    has_changes = True

    if has_changes:
        _commit_schema_update(client, schema)
    else:
        print("無任何更新變更。")


# ==========================================
# 核心功能: 刪除 (Delete)
# ==========================================


def delete_labels(
    project_id: str, location: str, processor_id: str, label_names: List[str]
):
    """
    刪除標籤。會從所有 Parent 的屬性中移除，並刪除 EntityType 定義。
    """
    client = get_dataset_client(location)
    schema, root_entity_type = _get_schema_resource(
        client, project_id, location, processor_id
    )
    if not schema:
        return

    has_changes = False

    for name_to_delete in label_names:
        print(f"   🗑️ 準備刪除: {name_to_delete}")

        deleted_count = 0

        # 1. 從所有 Entity Types 的 properties 中移除引用
        for et in schema.document_schema.entity_types:
            props_to_remove = [p for p in et.properties if p.name == name_to_delete]
            for p in props_to_remove:
                print(f"      ✂️ 從 Parent '{et.name}' 移除屬性參照")
                et.properties.remove(p)
                deleted_count += 1
                has_changes = True

        # 2. 移除 EntityType 定義本身 (如果存在)
        et_to_remove = next(
            (
                et
                for et in schema.document_schema.entity_types
                if et.name == name_to_delete
            ),
            None,
        )
        if et_to_remove:
            print(f"      ❌ 移除 EntityType 定義: {name_to_delete}")
            schema.document_schema.entity_types.remove(et_to_remove)
            deleted_count += 1
            has_changes = True

        if deleted_count == 0:
            print(f"      ⚠️ 未在 Schema 中找到 '{name_to_delete}'，無法刪除。")

    if has_changes:
        _commit_schema_update(client, schema)
    else:
        print("無任何刪除變更。")


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # --- 設定您的參數 ---
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = "us"
    PROCESSOR_ID = os.getenv("PROCESSOR_ID")

    print(f"🚀 開始管理 Processor Schema ({PROCESSOR_ID})...")

    # 範例 1: 新增
    add_list = [
        {
            "name": "pay_date",
            "type": "string",
            "description": "繳款日",
            "parent": "documents",
        },
        # {"name": "line_items", "type": "collection", "description": "明細行"}, # Collection Example
    ]
    if add_list:
        add_labels(PROJECT_ID, LOCATION, PROCESSOR_ID, add_list)

    # 範例 2: 更新
    update_list = [
        {"name": "invoice_id", "description": "發票號"},
    ]
    if update_list:
        update_labels(PROJECT_ID, LOCATION, PROCESSOR_ID, update_list)

    # 範例 3: 刪除
    delete_list = [
        "date",
    ]
    if delete_list:
        delete_labels(PROJECT_ID, LOCATION, PROCESSOR_ID, delete_list)

    # 最後顯示結果
    list_current_labels(PROJECT_ID, LOCATION, PROCESSOR_ID)

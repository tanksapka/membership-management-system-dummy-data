import pandas as pd
import uuid
from itertools import count
from typing import Dict, Iterable, Optional


def generate_uuid(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df.insert(len(df.columns), col, df.index.to_series().apply(lambda x: uuid.uuid1()))
    return df


def generate_mapping_dfs(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    relevant_keys = (
        'person', 'organization', 'gender', 'membership_fee_category', 'address_type', 'phone_type', 'email_type'
    )
    return {key: generate_uuid(df[[]].copy(), f'{key}_uuid') for key, df in df_dict.items() if key in relevant_keys}


def clean_up_id_columns(df: pd.DataFrame, indices: Optional[Iterable] = None) -> pd.DataFrame:
    if indices:
        column_order = iter(indices)
    else:
        column_order = count()
    col: str
    for col in df.columns:
        if col.find('uuid') != -1:
            df.insert(next(column_order), col, df.pop(col))
            df.drop(columns=[col.replace('uuid', 'id')], inplace=True)
            df.rename(columns={col: col.replace('uuid', 'id')}, inplace=True)
    return df


def map_id_fields(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in df_dict.items()}
    mappings = generate_mapping_dfs(df_dict)

    result['person'] = result['person'].merge(
        mappings['person'], how='left', left_index=True, right_index=True
    )
    result['person'].set_index('person_uuid', drop=True, inplace=True)
    result['person_data'] = result['person_data'].merge(
        mappings['person'], how='left', left_on='person_id', right_index=True
    )
    result['person_data'] = result['person_data'].merge(
        mappings['gender'], how='left', left_on='gender_id', right_index=True
    )
    result['person_data'] = result['person_data'].merge(
        mappings['membership_fee_category'], how='left', left_on='membership_fee_category_id', right_index=True
    )
    clean_up_id_columns(result['person_data'], (0, 7, 9))
    generate_uuid(result['person_data'], 'uuid')
    result['person_data'].set_index('uuid', drop=True, inplace=True)

    result['organization'] = result['organization'].merge(
        mappings['organization'], how='left', left_index=True, right_index=True
    )
    result['organization'].set_index('organization_uuid', drop=True, inplace=True)
    result['organization_data'] = result['organization_data'].merge(mappings['organization'], how='left',
                                                                    left_on='organization_id', right_index=True)
    result['organization_data'] = result['organization_data'].merge(mappings['organization'], how='left',
                                                                    left_on='organization_parent_id', right_index=True)
    result['organization_data'].rename(columns={
        'organization_uuid_x': 'organization_uuid',
        'organization_uuid_y': 'organization_parent_uuid',
    }, inplace=True)
    clean_up_id_columns(result['organization_data'])
    generate_uuid(result['organization_data'], 'uuid')
    result['organization_data'].set_index('uuid', drop=True, inplace=True)

    result['address'] = result['address'].merge(
        mappings['person'], how='left', left_on='person_id', right_index=True
    )
    result['address'] = result['address'].merge(
        mappings['organization'], how='left', left_on='organization_id', right_index=True
    )
    result['address'] = result['address'].merge(
        mappings['address_type'], how='left', left_on='address_type_id', right_index=True
    )
    clean_up_id_columns(result['address'])
    generate_uuid(result['address'], 'uuid')
    result['address'].set_index('uuid', drop=True, inplace=True)

    result['phone'] = result['phone'].merge(
        mappings['person'], how='left', left_on='person_id', right_index=True
    )
    result['phone'] = result['phone'].merge(
        mappings['organization'], how='left', left_on='organization_id', right_index=True
    )
    result['phone'] = result['phone'].merge(
        mappings['phone_type'], how='left', left_on='phone_type_id', right_index=True
    )
    clean_up_id_columns(result['phone'])
    generate_uuid(result['phone'], 'uuid')
    result['phone'].set_index('uuid', drop=True, inplace=True)

    result['email'] = result['email'].merge(
        mappings['person'], how='left', left_on='person_id', right_index=True
    )
    result['email'] = result['email'].merge(
        mappings['organization'], how='left', left_on='organization_id', right_index=True
    )
    result['email'] = result['email'].merge(
        mappings['email_type'], how='left', left_on='email_type_id', right_index=True
    )
    clean_up_id_columns(result['email'])
    generate_uuid(result['email'], 'uuid')
    result['email'].set_index('uuid', drop=True, inplace=True)

    result['membership'] = result['membership'].merge(
        mappings['person'], how='left', left_on='person_id', right_index=True
    )
    result['membership'] = result['membership'].merge(
        mappings['organization'], how='left', left_on='organization_id', right_index=True
    )
    clean_up_id_columns(result['membership'])
    generate_uuid(result['membership'], 'uuid')
    result['membership'].set_index('uuid', drop=True, inplace=True)

    result['gender'] = result['gender'].merge(
        mappings['gender'], how='left', left_index=True, right_index=True
    )
    result['gender'].set_index('gender_uuid', drop=True, inplace=True)

    result['membership_fee_category'] = result['membership_fee_category'].merge(
        mappings['membership_fee_category'], how='left', left_index=True, right_index=True
    )
    result['membership_fee_category'].set_index('membership_fee_category_uuid', drop=True, inplace=True)

    result['address_type'] = result['address_type'].merge(
        mappings['address_type'], how='left', left_index=True, right_index=True
    )
    result['address_type'].set_index('address_type_uuid', drop=True, inplace=True)

    result['phone_type'] = result['phone_type'].merge(
        mappings['phone_type'], how='left', left_index=True, right_index=True
    )
    result['phone_type'].set_index('phone_type_uuid', drop=True, inplace=True)

    result['email_type'] = result['email_type'].merge(
        mappings['email_type'], how='left', left_index=True, right_index=True
    )
    result['email_type'].set_index('email_type_uuid', drop=True, inplace=True)

    return result

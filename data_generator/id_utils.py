import pandas as pd
import uuid
from typing import Dict

RELEVANT_FIELDS = {'id', 'person_id', 'organization_id'}


def map_id_fields(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in df_dict.items()}
    mappings = dict()

    mappings['person'] = df_dict['person'][[]].copy()
    mappings['person'].insert(len(mappings['person'].columns), 'uuid', uuid.uuid1())
    mappings['organization'] = df_dict['organization'][[]].copy()
    mappings['organization'].insert(len(mappings['organization'].columns), 'uuid', uuid.uuid1())
    mappings['gender'] = df_dict['gender'][[]].copy()
    mappings['gender'].insert(len(mappings['gender'].columns), 'uuid', uuid.uuid1())
    mappings['membership_fee_category'] = df_dict['membership_fee_category'][[]].copy()
    mappings['membership_fee_category'].insert(len(mappings['membership_fee_category'].columns), 'uuid', uuid.uuid1())
    mappings['address_type'] = df_dict['address_type'][[]].copy()
    mappings['address_type'].insert(len(mappings['address_type'].columns), 'uuid', uuid.uuid1())
    mappings['phone_type'] = df_dict['phone_type'][[]].copy()
    mappings['phone_type'].insert(len(mappings['phone_type'].columns), 'uuid', uuid.uuid1())
    mappings['email_type'] = df_dict['email_type'][[]].copy()
    mappings['email_type'].insert(len(mappings['email_type'].columns), 'uuid', uuid.uuid1())

    result['person_data'] = result['person_data'].merge(mappings['person'], how='left', left_on='person_id',
                                                        right_index=True)
    result['person_data'] = result['person_data'].merge(mappings['gender'], how='left', left_on='gender_id',
                                                        right_index=True, suffixes=('_gender', '_gender'))
    result['person_data'] = result['person_data'].merge(mappings['membership_fee_category'], how='left',
                                                        left_on='membership_fee_category_id', right_index=True,
                                                        suffixes=('_membership_fee_category', '_membership_fee_category'))
    result['organization_data'] = result['organization_data'].merge(mappings['organization'], how='left',
                                                                    left_on='organization_id', right_index=True)

    return result

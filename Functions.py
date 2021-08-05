import numpy as np
import pandas as pd
import pickle

def preprocess_application(application):
    '''function for cleaning and feature engineering of application_train.csv and application_test.csv'''

    # Remove missing values in GENDER
    application = application[application['CODE_GENDER'] != 'XNA']

    # Create an anomalous flag column
    application['DAYS_EMPLOYED_ANOM'] = application["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    application['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # Drop non relevant cols (<50 rows with different values)
    cols_to_drop = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12',
                            'FLAG_DOCUMENT_20']
    application = application.drop(cols_to_drop, axis = 1)

    # Fill missing values in categorical columns with 'XNA' value
    categorical_columns = application.dtypes[application.dtypes == 'object'].index.tolist()
    application[categorical_columns] = application[categorical_columns].fillna('XNA')

    # Encode ordinal categorical columns
    binary_dict = { "Y" : 1, "N" : 0}
    application['FLAG_OWN_CAR'] = application['FLAG_OWN_CAR'].map(binary_dict)
    application['FLAG_OWN_REALTY'] = application['FLAG_OWN_REALTY'].map(binary_dict)

    week_dict = {'MONDAY':0, 'TUESDAY':1, 'WEDNESDAY':2, 'THURSDAY':3, 'FRIDAY':4, 'SATURDAY':5, 'SUNDAY':6}
    application['WEEKDAY_APPR_PROCESS_START'] = application['WEEKDAY_APPR_PROCESS_START'].map(week_dict)

    education_dict = {'Lower secondary': 0, 'Secondary / secondary special':1,
                  'Incomplete higher':2, 'Higher education':3, 'Academic degree':4  }
    application['NAME_EDUCATION_TYPE'] = application['NAME_EDUCATION_TYPE'].map(education_dict)
    
    # drop columns that are more than 40% Nan and less correlation to target variable 
    columns_to_drop = ['APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG',
                       'COMMONAREA_AVG','ELEVATORS_AVG',
    'ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE',
    'YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE',
    'LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI',
    'ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI',
    'LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
    'TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']

    # create new df for FE
    application_fe = application.copy().drop(columns = columns_to_drop, axis = 1)
    application_fe.shape
    
    # the percentage of the credit amount relative to a client's income
    application_fe['CREDIT_INCOME_RATIO'] = application_fe['AMT_CREDIT'] / (application_fe['AMT_INCOME_TOTAL'] + 0.00001)

    # Ratio and difference between loan annuity and client's income
    application_fe['ANNUITY_INCOME_RATIO'] = application_fe['AMT_ANNUITY'] / (application_fe['AMT_INCOME_TOTAL']+ 0.00001)
    application_fe['INCOME_ANNUITY_DIFF'] = application_fe['AMT_INCOME_TOTAL'] - application_fe['AMT_ANNUITY']
        
    # Ratio and difference between amount credit and amount goods
    application_fe['CREDIT_GOODS_RATIO'] = application_fe['AMT_CREDIT'] / (application_fe['AMT_GOODS_PRICE'] + 0.00001)
    application_fe['CREDIT_GOODS_DIFF'] = application_fe['AMT_CREDIT'] - application_fe['AMT_GOODS_PRICE'] + 0.00001

    # the length of the payment in months
    application_fe['CREDIT_TERM'] = application_fe['AMT_CREDIT'] /(application_fe['AMT_ANNUITY']+ 0.00001)

    # the percentage of the days employed relative to the client's age
    application_fe['DAYS_EMPLOYED_RATIO'] = application_fe['DAYS_EMPLOYED'] / (application_fe['DAYS_BIRTH']+ 0.00001)

    # flag contacts sum
    application_fe['FLAG_CONTACTS_SUM'] = application_fe['FLAG_MOBIL'] + application_fe['FLAG_EMP_PHONE']\
                                          + application_fe['FLAG_WORK_PHONE'] + application_fe['FLAG_CONT_MOBILE']\
                                          + application_fe['FLAG_PHONE'] + application_fe['FLAG_EMAIL']

    # family members
    application_fe['CNT_NON_CHILDREN'] = application_fe['CNT_FAM_MEMBERS'] - application_fe['CNT_CHILDREN']
    application_fe['CHILDREN_INCOME_RATIO'] = application_fe['CNT_CHILDREN'] / (application_fe['AMT_INCOME_TOTAL'] + 0.00001)
    application_fe['PER_CAPITA_INCOME'] = application_fe['AMT_INCOME_TOTAL'] / (application_fe['CNT_FAM_MEMBERS'] + 1)

    # flag regions
    application_fe['FLAG_REGIONS'] = application_fe['REG_REGION_NOT_LIVE_REGION'] + application_fe['REG_REGION_NOT_WORK_REGION'] \
                                         + application_fe['LIVE_REGION_NOT_WORK_REGION']+application_fe['REG_CITY_NOT_LIVE_CITY'] \
                                         + application_fe['REG_CITY_NOT_WORK_CITY'] + application_fe['LIVE_CITY_NOT_WORK_CITY']   

    # sum flag documents
    application_fe['SUM_FLAGS_DOCUMENTS'] = application_fe['FLAG_DOCUMENT_3']\
    + application_fe['FLAG_DOCUMENT_5'] + application_fe['FLAG_DOCUMENT_6']\
    + application_fe['FLAG_DOCUMENT_7'] + application_fe['FLAG_DOCUMENT_8'] + application_fe['FLAG_DOCUMENT_9']\
    + application_fe['FLAG_DOCUMENT_11'] + application_fe['FLAG_DOCUMENT_13'] + application_fe['FLAG_DOCUMENT_14']\
    + application_fe['FLAG_DOCUMENT_15'] + application_fe['FLAG_DOCUMENT_16'] + application_fe['FLAG_DOCUMENT_17']\
    + application_fe['FLAG_DOCUMENT_18'] + application_fe['FLAG_DOCUMENT_19'] + application_fe['FLAG_DOCUMENT_21']

    return(application_fe)



def preprocess_bureau(bureau):
    '''function for cleaning and featue engineering of bureau.csv'''

    # Assign NaN to outliers for 'DAYS_CREDIT_ENDDATE', DAYS_CREDIT_UPDATE & 'DAYS_ENDDATE_FACT'
    # only keep loans within the last 50 years
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] > -
               50*365, 'DAYS_CREDIT_ENDDATE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] > -
               50*365, 'DAYS_ENDDATE_FACT'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_UPDATE'] > -
               50*365, 'DAYS_CREDIT_UPDATE'] = np.nan

    # https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering
    # create binary variable : 1 if CREDIT_ACTIVE status is 'Active', 0 otherwise
    bureau['CREDIT_ACTIVE_BINARY'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Active', 1, 0)

    # create binary variable : 1 if 'DAYS_CREDIT_ENDDATE' is <0 (credit end date was in the past
    # at the time of the application), 0 otherwise ( credit end date in the future)
    bureau['DAYS_CREDIT_ENDDATE_BINARY'] = np.where(
        (bureau['DAYS_CREDIT_ENDDATE'] < 0) & (bureau['CREDIT_ACTIVE_BINARY'] == 1), 1, 0)

    # create binary variable : 1 if CREDIT_DAY_OVERDUE is not 0
    bureau['CREDIT_DAY_OVERDUE_BINARY'] = np.where(
        bureau['CREDIT_DAY_OVERDUE'] != 0, 1, 0)

    # Aggregate df by 'SK_ID_CURR' for feature engineering
    bureau_fe = bureau.groupby('SK_ID_CURR').agg(
        # number of loans with Bureau credit
        BUREAU_COUNT=('SK_ID_BUREAU', 'count'),
        # number of different types of loans
        BUREAU_TYPES_COUNT=('CREDIT_TYPE', 'nunique'),
        # percentage of active loans
        BUREAU_ACTIVE_LOANS_PCT=('CREDIT_ACTIVE_BINARY', 'mean'),
        # percentage of active loans with end date in the past
        BUREAU_PAST_DUE_LOANS_PCT=('DAYS_CREDIT_ENDDATE_BINARY', 'mean'),
        # sum of debts for all loans
        BUREAU_TOTAL_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
        # sum of credits for all loans
        BUREAU_TOTAL_CREDIT=('AMT_CREDIT_SUM', 'sum'),
        # sum of overdue payments for all loans
        BUREAU_TOTAL_OVERDUE=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        # average number of time the credit was prolonged
        BUREAU_CREDIT_PROL_AVG=('CNT_CREDIT_PROLONG', 'mean'),
        BUREAU_OVERDUE_COUNT=('CREDIT_DAY_OVERDUE_BINARY', 'sum'),
        BUREAU_DAYS_DIFF_AVG=('DAYS_CREDIT',  # average difference of days between each loans
                              lambda x: x.sort_values().diff().fillna(0).mean())
    )

    # Average number of types of loans
    bureau_fe['BUREAU_AVG_TYPES_COUNT'] = bureau_fe['BUREAU_COUNT'] / \
        bureau_fe['BUREAU_TYPES_COUNT']

    # Ratio total debt / total credit
    bureau_fe['BUREAU_RATIO_DEBT_CREDIT'] = np.where(bureau_fe['BUREAU_TOTAL_CREDIT'] != 0,
                                                     bureau_fe['BUREAU_TOTAL_DEBT'] / bureau_fe['BUREAU_TOTAL_CREDIT'], 0)
    # Ratio total overdue / total debt
    bureau_fe['BUREAU_RATIO_OVERDUE_DEBT'] = np.where(bureau_fe['BUREAU_TOTAL_DEBT'] != 0,
                                                      bureau_fe['BUREAU_TOTAL_OVERDUE'] / bureau_fe['BUREAU_TOTAL_DEBT'], 0)
    #bureau_fe = bureau_fe.reset_index()
    return(bureau_fe)


def preprocess_prevapp(prev_application):
    '''function for cleaning and feature engineering of previous_application.csv'''
 
    # label encoding the categorical variables
    name_contract_dict = {'Approved': 0, 'Refused': 3,
                          'Canceled': 2, 'Unused offer': 1}
    prev_application['NAME_CONTRACT_STATUS'] = prev_application['NAME_CONTRACT_STATUS'].map(
        name_contract_dict)

    yield_group_dict = {'XNA': 0, 'low_action': 1,
                        'low_normal': 2, 'middle': 3, 'high': 4}
    prev_application['NAME_YIELD_GROUP'] = prev_application['NAME_YIELD_GROUP'].map(
        yield_group_dict)

    appl_per_contract_last_dict = {'Y': 1, 'N': 0}
    prev_application['FLAG_LAST_APPL_PER_CONTRACT'] = prev_application['FLAG_LAST_APPL_PER_CONTRACT'].map(
        appl_per_contract_last_dict)

    # engineering some features on domain knowledge
    prev_application['AMT_DECLINED'] = prev_application['AMT_APPLICATION'] - \
        prev_application['AMT_CREDIT']
    prev_application['AMT_CREDIT_GOODS_RATIO'] = prev_application['AMT_CREDIT'] / \
        (prev_application['AMT_GOODS_PRICE'] + 0.00001)
    prev_application['AMT_CREDIT_GOODS_DIFF'] = prev_application['AMT_CREDIT'] - \
        prev_application['AMT_GOODS_PRICE']
    prev_application['AMT_CREDIT_APPLICATION_RATIO'] = prev_application['AMT_APPLICATION'] / \
        (prev_application['AMT_CREDIT'] + 0.00001)
    prev_application['CREDIT_DOWNPAYMENT_RATIO'] = prev_application['AMT_DOWN_PAYMENT'] / \
        (prev_application['AMT_CREDIT'] + 0.00001)
    prev_application['GOOD_DOWNPAYMET_RATIO'] = prev_application['AMT_DOWN_PAYMENT'] / \
        (prev_application['AMT_GOODS_PRICE'] + 0.00001)
    prev_application['ANNUITY'] = prev_application['AMT_CREDIT'] / \
        (prev_application['CNT_PAYMENT'] + 0.00001)
    prev_application['ANNUITY_GOODS'] = prev_application['AMT_GOODS_PRICE'] / \
        (prev_application['CNT_PAYMENT'] + 0.00001)
    prev_application['DAYS_FIRST_LAST_DUE_DIFF'] = prev_application['DAYS_LAST_DUE'] - \
        prev_application['DAYS_FIRST_DUE']

    aggregations_for_prev_application = {
        'SK_ID_PREV': ['count'],
        'AMT_ANNUITY': ['last'],
        'AMT_APPLICATION': ['last'],
        'AMT_CREDIT': ['last'],
        'AMT_DOWN_PAYMENT': ['last'],
        'AMT_GOODS_PRICE': ['last'],
        'FLAG_LAST_APPL_PER_CONTRACT': ['last'],
        'DAYS_FIRST_DUE': ['last'],
        'DAYS_LAST_DUE_1ST_VERSION': ['last'],
        'DAYS_LAST_DUE': ['last'],
        'AMT_DECLINED': ['last'],
        'AMT_CREDIT_GOODS_RATIO': ['last'],
        'AMT_CREDIT_GOODS_DIFF': ['last'],
        'AMT_CREDIT_APPLICATION_RATIO': ['last'],
        'CREDIT_DOWNPAYMENT_RATIO': ['last'],
        'GOOD_DOWNPAYMET_RATIO': ['last'],
        'ANNUITY': ['last'],
        'ANNUITY_GOODS': ['last']}

    # performing overall aggregations over SK_ID_PREV for all features
    prev_application_fe = prev_application.groupby(
        'SK_ID_CURR').agg(aggregations_for_prev_application)
    cols = ['_'.join(ele).upper() for ele in prev_application_fe.columns]
    prev_application_fe.columns = ['PREVAPP_' + ele for ele in cols]
    
    #prev_application_fe = prev_application_fe.reset_index()

    return prev_application_fe


def merge_data(application_fe, bureau_fe, prev_application_fe):
    '''function to merge the 3 df after cleaning and feature engineering'''
    data = application_fe\
        .merge(bureau_fe, on='SK_ID_CURR', how='left')\
        .merge(prev_application_fe, on='SK_ID_CURR', how='left')

    return data


def preprocessing_data(data):
    '''function to preprocess final data and return a df with 35 selected variables'''
    
    categorical_var = [col for col in data.columns if data.dtypes[col] == 'object']
    numerical_var = [col for col in data.columns if (col not in categorical_var) and (col != 'TARGET') and (col != 'SK_ID_CURR')]

    # Numerical variables
    preprocess_num_var = pickle.load(open('preprocess_numerical.pkl', 'rb'))
    data_num_pp = pd.DataFrame(preprocess_num_var.transform(data[numerical_var]), columns=numerical_var)

    # preprocessing categorical variables
    selected_cat = ['CODE_GENDER', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE']
    selected_cat_features = ['CODE_GENDER_F', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_XNA',
                             'NAME_INCOME_TYPE_Working']
    # encode categorical variables
    data_cat_pp = pd.get_dummies(data[selected_cat])

    for var in ['CODE_GENDER_F', 'OCCUPATION_TYPE_XNA', 'NAME_INCOME_TYPE_Working', 'OCCUPATION_TYPE_Laborers']:
        if var not in data_cat_pp.columns:
            data_cat_pp[var] = 0

    # filter on selected variables
    data_cat_pp = data_cat_pp[selected_cat_features]

    # concat
    selected_variables = ['DAYS_BIRTH', 'INCOME_ANNUITY_DIFF', 'PREVAPP_AMT_ANNUITY_LAST',
                          'ANNUITY_INCOME_RATIO', 'FLAG_EMP_PHONE', 'BUREAU_TOTAL_DEBT',
                          'REGION_POPULATION_RELATIVE', 'REG_CITY_NOT_LIVE_CITY', 'CODE_GENDER_F',
                          'REGION_RATING_CLIENT_W_CITY', 'PREVAPP_AMT_CREDIT_GOODS_DIFF_LAST',
                          'REG_CITY_NOT_WORK_CITY', 'BUREAU_RATIO_DEBT_CREDIT',
                          'DAYS_LAST_PHONE_CHANGE', 'EXT_SOURCE_1', 'PREVAPP_SK_ID_PREV_COUNT',
                          'CREDIT_GOODS_RATIO', 'BUREAU_COUNT', 'NAME_EDUCATION_TYPE',
                          'OCCUPATION_TYPE_Laborers', 'AMT_GOODS_PRICE', 'OCCUPATION_TYPE_XNA',
                          'EXT_SOURCE_2', 'DAYS_EMPLOYED_RATIO', 'PREVAPP_AMT_DECLINED_LAST',
                          'DAYS_REGISTRATION', 'PREVAPP_ANNUITY_GOODS_LAST', 'FLAG_DOCUMENT_3',
                          'BUREAU_DAYS_DIFF_AVG', 'DAYS_ID_PUBLISH', 'BUREAU_ACTIVE_LOANS_PCT',
                          'PREVAPP_DAYS_LAST_DUE_1ST_VERSION_LAST', 'NAME_INCOME_TYPE_Working',
                          'BUREAU_TOTAL_CREDIT', 'CREDIT_TERM']

    data_pp = pd.concat([data_num_pp, data_cat_pp], axis=1)[selected_variables]
    return (data_pp)





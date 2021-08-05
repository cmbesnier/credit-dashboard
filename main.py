# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import SessionState
import pickle
import lightgbm
from Functions import *

# picture and title
col1, mid, col2 = st.beta_columns([5, 1, 15])
with col1:
    path = "data/picture.png"
    st.image(path, width=150)

with col2:
    st.title('Bank Loan Dashboard')
    st.write('Credit Default Risk Prediction')

# Load data
def load_data():
    application = pd.read_csv('data/application_test_sample.csv', index_col=0)
    bureau = pd.read_csv('data/bureau_test_fe.csv', index_col=0)
    prev_application = pd.read_csv('data/prev_application_test_fe.csv', index_col=0)
    data_compare_clients = pd.read_csv('data/data_compare_clients.csv')

    return application, bureau, prev_application, data_compare_clients

application, bureau, prev_application, data_compare_clients = load_data()

# Select a sample of 100 clients
application_sample = application.sample(100, random_state = 0)
client_list = application_sample['SK_ID_CURR'].tolist()

# Prepare data for this set of clients
application_sample_fe = preprocess_application(application_sample)
bureau_sample = bureau[bureau['SK_ID_CURR'].isin(client_list)]
prev_application_sample = prev_application[prev_application['SK_ID_CURR'].isin(client_list)]
data = merge_data(application_sample_fe, bureau_sample, prev_application_sample)
data = data.fillna(data.median())

# Select one client ID
col1, col2 = st.sidebar.beta_columns([2,1])
with col1:
    st.title('Client Selection')


col1, col2, col3 = st.sidebar.beta_columns([2,1,1])
with col1:
    select_client = st.selectbox('Select a Client', client_list)

with col3:
    # Set session counter for reset option
    session = SessionState.get(run_id=0)
    if st.button("Reset"):
        session.run_id += 1

# Set labels for variables
name_dict = {'NAME_INCOME_TYPE': 'Income Type',
             'NAME_EDUCATION_TYPE': 'Education Type',
             'NAME_FAMILY_STATUS': 'Family Status',
             'CNT_CHILDREN': "Nb Children",
             'AMT_INCOME_TOTAL': 'Total Income',
             'AMT_CREDIT': 'Credit Amount',
             'AMT_ANNUITY': 'Annuity',
             'BUREAU_COUNT': 'Bureau Credit Nb Loans',
             'BUREAU_ACTIVE_LOANS_PCT': 'Bureau Credit  % Active Loans',
             'PREVAPP_SK_ID_PREV_COUNT': 'Nb Previous Application',
             }


# Store initial values
init_dict = dict()
for var in name_dict.keys():
    init_dict[var] = data.loc[data['SK_ID_CURR'] == select_client, var].max()

# Map education name
education_dict = {0 : 'Lower secondary', 1:  'Secondary / secondary special',
                  2: 'Incomplete higher', 3: 'Higher education', 4: 'Academic degree' }
init_dict['NAME_EDUCATION_TYPE'] = education_dict[init_dict['NAME_EDUCATION_TYPE']]

# display client parameters
st.markdown('#')
st.markdown('** Client Details **')

# Categorical variables (static)
col1, col2, col3, col4 = st.beta_columns(4)
cols = [col1, col2, col3, col4]
for i, var in enumerate(['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN']):
        with cols[i]:
            st.markdown(name_dict[var])
            st.info(init_dict[var])

col1, col2, col3, col4 = st.beta_columns(4)
cols = [col1, col2, col3]
for i, var in enumerate(['BUREAU_COUNT', 'BUREAU_ACTIVE_LOANS_PCT', 'PREVAPP_SK_ID_PREV_COUNT']):
        with cols[i]:
            st.markdown(name_dict[var])
            st.info(int(init_dict[var]))


# Numerical variables with sliders
select_dict = dict()
col5, col6, col7, col8 = st.beta_columns(4)
cols = [col5, col6, col7, col8]
keys = [0, 10, 20, 30, 40]
for i, var in enumerate(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']):
    with cols[i]:
        select_dict[var] = st.slider(name_dict[var],
                                int(0),
                                int(round(data[var].max(), -3)),
                                int(init_dict[var]),
                                key=str(keys[i] + session.run_id)
                                )

# Display model result
st.markdown('#')
st.markdown('** Scoring **')

# load client's data
X = data[data['SK_ID_CURR'] == select_client]

# update slider values in client's data
for var in select_dict.keys():
    X.loc[X['SK_ID_CURR']==select_client, var] = select_dict[var]
X['CREDIT_TERM'] = select_dict['AMT_CREDIT'] /(select_dict['AMT_ANNUITY']+ 0.00001)
X['ANNUITY_INCOME_RATIO'] = select_dict['AMT_ANNUITY'] / (select_dict['AMT_INCOME_TOTAL']+ 0.00001)
X['INCOME_ANNUITY_DIFF'] = select_dict['AMT_INCOME_TOTAL'] - select_dict['AMT_ANNUITY']
X['CREDIT_GOODS_RATIO'] = select_dict['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 0.00001)

# Apply preprocessing
X_pp = preprocessing_data(X)

# load model
model = pickle.load(open('lgbm_model.pkl', 'rb'))
prevision = model.predict_proba(X_pp)[0,1]

# Set threshold
default_threshold = 0.516
col1, col2, col3 = st.beta_columns((2,1,1))
with col2:
    threshold = st.slider('Threshold',0.0, 1.0, default_threshold, key=str(keys[4] + session.run_id))

# display score and risk proba
with col1:
    if prevision < threshold:
        st.success('CREDIT GRANTED')
    elif prevision >= threshold:
        st.warning('CREDIT DENIED')

    st.write("Default Risk : {} %".format(round(prevision * 100)))

# display comparison graph
st.sidebar.title('Statistics')

# Select pop to compare to
col1, col2 = st.sidebar.beta_columns([1,1])
with col1:
    pop = st.selectbox('Compare Client to', ['All Clients', 'Non Defaulters', 'Defaulters'])


# Select a variable
param_name_dict = {
                      'External Source 2': 'EXT_SOURCE_2',
                      'Credit Term': 'CREDIT_TERM',
                      'External Source 1': 'EXT_SOURCE_1',
                      'Credit Goods Ratio': 'CREDIT_GOODS_RATIO',
                      'Bureau Active Loan %': 'BUREAU_ACTIVE_LOANS_PCT',
                      'Days Employed Ratio': 'DAYS_EMPLOYED_RATIO',
                      'Bureau Debt Credit Ratio': 'BUREAU_RATIO_DEBT_CREDIT',
                      'Age': 'DAYS_BIRTH',
                      'Amount Goods Price': 'AMT_GOODS_PRICE',
                        'Annuity Income Ratio':'ANNUITY_INCOME_RATIO',
}

param_list = list(param_name_dict.keys())

# get selected variable
param = st.sidebar.selectbox('Select Parameter', param_list, 7)

# get client value for this variable
x = data[data['SK_ID_CURR'] == select_client][param_name_dict[param]].item()

if param == 'Age':
    x = -x/365

# Displot of selected variable
df = data_compare_clients.sample(500, random_state = 0)
df = df.fillna(df.median())
df['DAYS_BIRTH'] = -df['DAYS_BIRTH']/365

pop_dict = {'Defaulters' : 1,
            'Non Defaulters':0
            }

if pop != 'All Clients':
    fig, ax = plt.subplots(figsize = (4,3))
    sns.histplot(x = param_name_dict[param], data = df[df['TARGET']== pop_dict[pop]],
             kde = True, linewidth=0,
             color = 'darkgreen', bins = 30, ax = ax, alpha = 0.3)
    mean = df[df['TARGET']== pop_dict[pop]][param_name_dict[param]].mean()
else :
    fig, ax = plt.subplots(figsize = (4,3))
    sns.histplot(x = param_name_dict[param], data = df,
             kde = True, linewidth=0,
             color = 'darkgreen', bins = 30, ax = ax, alpha = 0.3)
    mean = df[param_name_dict[param]].mean()
plt.xlabel(param)

# indicate client data
plt.axvline(x= x, c = '#F37768')
fig.patch.set_alpha(0)
st.sidebar.pyplot(fig)

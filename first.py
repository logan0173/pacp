import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

st.markdown("<h1 style='text-align: center; color: green;'>PACP Prediction</h1>", unsafe_allow_html=True)

#st.write("""
# PACP Prediction App """)

st.write("""
## Enter pipe details below:
""")
    
a = st.number_input(' Age: ', min_value=0)
l = st.number_input(" Length: ", min_value=0)
s = st.number_input(" Slope: ", min_value=0.000)
d = st.number_input(" Diameter: ", min_value=0)

st.info(
    "**AT: 0, CI: 1, Concrete: 2, DI: 3, Fiberglass: 4, HDPE: 5, PVC: 6, Steel: 7, Clay: 8**")

sel_col, disp_col= st.beta_columns(2)
m = sel_col.slider('Material: ', min_value=0, max_value=8, value=6, step=1)

suba = st.text_input(" SubArea: ")
mapsc = st.text_input(" MapscoGrid reference: ")

df = pd.read_csv('encoded data.csv')
df_sub = pd.read_csv('subarea.csv')
df_sub.head()
df_map = pd.read_csv('mapsco.csv')
df_map.head()
    
sub = df_sub.loc[df_sub['SUBAREA'] == suba, 'SUBAREA_Encoded']
maps = df_map.loc[df_map['MAPSCOGRID'] == mapsc, 'MAPSCOGRID_Encoded']
    
X = df.drop(['Unnamed: 0','PACP','MAPSCOGRID','SUBAREA','MATERIAL'],axis=1)
y = df['PACP']
    
ros = RandomOverSampler(random_state=0)
X_over, y_over = ros.fit_resample(X, y)
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_over, y_over, test_size=0.2, shuffle=True, random_state=0)
    
clf_rf = RandomForestClassifier(random_state=0)
model = clf_rf.fit(X_train_over, y_train_over) 

PACP = model.predict([[a,l,s,d,sub,m,maps]])

if PACP==5:
    st.error("# PACP Score is: {}".format(PACP))
    st.error("# Pipe failed/likely to fail; Inspect Now!")
elif PACP==4:
    st.warning("# PACP Score is: {}".format(PACP))
    st.warning("# Inspection Required in 5 years!")
elif PACP==3:
    st.info("# PACP Score is: {}".format(PACP))
    st.info("# Pipe is in fair condition; Inspection required after 10 years!")
else:
    st.success("# PACP Score is: {}".format(PACP))
    st.success("# Pipe is in good shape; Inspection not required!")
    


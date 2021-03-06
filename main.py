import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('per capita crime rate by town', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('proportion of residential land zoned for lots over 25,000 sq.ft.', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('proportion of non-retail business acres per town', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('Charles River dummy variable (1 if tract bounds river; 0 otherwise)', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('nitric oxides concentration (parts per 10 million)', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('average number of rooms per dwelling', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('proportion of owner-occupied units built prior to 1940', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('weighted distances to five Boston employment centres', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('index of accessibility to radial highways', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('full-value property-tax rate per $10,000', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('pupil-teacher ratio by town', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('% lower status of the population', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Distribution of House Price Data
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Target Data Distribution')
plt.title('Distribution of House Price Data')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(Y['MEDV'], bins=30)
plt.show()
st.pyplot(bbox_inches='tight')
st.write('---')

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = linear_model.LinearRegression()
# train test split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=42)

# Train the model
model.fit(x_train,y_train)

# Apply Model to Make Prediction
y_pred =model.predict(x_test)

y_pred_df=pd.DataFrame(model.predict(df))
y_pred_df.rename(columns = {0:'Predicted Median value of owner-occupied homes in $1000s'}, inplace = True)

st.header('Prediction of MEDV')
st.write(y_pred_df)
st.write('---')

# Model Accuracy
st.header('Model Accuracy Metrics')
# The mean squared error
st.write("MSE: %.2f" % mean_squared_error(y_test,y_pred))
st.write(f'RMSE: {np.sqrt(mean_squared_error(y_test,y_pred))}')
# The coefficient of determination: 1 is perfect prediction
st.write("Coefficient of determination: %.2f" % r2_score(y_test,y_pred))

# Plot outputs
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Actual vs Predicted Plot')
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices ($1000)")
plt.ylabel("Predicted House Prices: ($1000)")
plt.title("Actual Prices vs Predicted prices")
st.pyplot(bbox_inches='tight')
st.write('---')

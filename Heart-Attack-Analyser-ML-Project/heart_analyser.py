import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import textwrap
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA

import os,sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import pyotp
import pandas as pd
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


# In[2]:
start_time=datetime.datetime.now()

heart=r'heart.csv'


data=pd.read_csv(heart)

data=data.fillna(data.mean())

X=data.drop(["result"],axis=1)
Y=data['result']

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("result", axis=1),
                                                    data["result"],
                                                    test_size=0.3)

def get_classifier(clf_name):
    global clf
    if clf_name == "Logistic Regression":
        clf = LogisticRegression()

    elif clf_name == "KNN":
        clf = KNeighborsClassifier()

    elif clf_name == "SVM":
        clf = SVC()

    elif clf_name == "Decision Trees":
        clf = DecisionTreeClassifier()
        
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier()
        
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier()
        
    elif clf_name == "XGBoost":
        clf = XGBClassifier()        
    return clf


# In[13]:


def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.23,random_state=65)

    #MinMax Scaling / Normalization of data
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)

    return X_train, X_test, Y_train, Y_test,Y_pred, Y_test,acc



clf_list=["Logistic Regression","KNN","SVM","Decision Trees","Random Forest","Gradient Boosting","XGBoost"]


fake_acc=0

for i in reversed(range(len(clf_list))):
    clf=get_classifier(clf_list[i])
    X_train, X_test, Y_train, Y_test,Y_pred, Y_test,acc=model()
    if acc>fake_acc:
        fake_acc=acc
        index=i 
        
clf=get_classifier(clf_list[index])
X_train, X_test, Y_train, Y_test,Y_pred, Y_test,acc=model()


def report():
    user=pd.read_csv("user_input.csv")
    user=user.columns.to_list()
    if index==0:
        for i in range(len(user)):
            aa=float(user[i])
            user[i]=int(aa)
    user=[user]

    # Convert input data to a DataFrame
    input_df = pd.DataFrame(user, columns=['Age', 'Sex(0-F;1-M)', 'Chest Pain',
       'Resting Blood Pressure (in mm hg)', 'Cholestrol',
       'fasting blood sugar (1 = true; 0 = false)',
       'Resting electrocardiographic results', 'Thalachh',
       'Exercise induced angina (1 = yes; 0 = no)', 'oldpeak',
        'Coronay artery anomalies',
       'Thalassemia'])
    # Predict the value using the model
    predicted_value = clf.predict(input_df)
#     predicted_value=predicted_value[0]
    return predicted_value,input_df


def genrate_report():
    a,df=report()
    text1=str(df.to_dict())
    text2=' fully analyse this report, explain everything and what is the precaution for report, this is person heart data analyse report and tell me what precaustion he/she should take for their health, and tell if there is any thing wrong in the report'
    text=text1+text2
    

    # Create a new instance of the Chrome driver
    # driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    chromedriver_path = r"C:\Users\Pranjali PC\.cache\selenium\chromedriver\win64\136.0.7103.94\chromedriver.exe"

    service = ChromeService(executable_path=chromedriver_path)
    options = webdriver.ChromeOptions()

    driver = webdriver.Chrome(service=service, options=options)

    # Open copilot webpage
    driver.get('https://copilot.microsoft.com/')

    
    time.sleep(5)  # Adjust this delay as needed

    # Find the chat input field by class name

    input_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="userInput"]'))
    )
    # Input text into the element
    input_element.send_keys(text)


    time.sleep(1)
    input_element.send_keys(Keys.ENTER)

    time.sleep(45)
    full_xpath = '/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div/div/div/div[2]/div[2]/div[2]/div[3]/div'

    response_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, full_xpath)))

    # Extract the text of the response
    value = response_element.text

    value = value.replace('. ', '.\n \n')
    driver.quit()
    return value

end_time=datetime.datetime.now()
print(end_time-start_time)
# %%

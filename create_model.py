import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Dataset load kar
df = pd.read_csv('loan_train.csv') # ← Tera dataset ka naam daal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Data load karo
df = pd.read_csv('loan_train.csv')

# 2. NAYI LINES - NaN hatane ke liye 👇
df = df.dropna() # Sab khali rows delete kar dega
print(f"NaN hatane ke baad rows: {len(df)}")

# 3. Baaki tera code same rahega...
# df = df.drop('Loan_ID', axis=1) # Agar ye line hai to rehne de
#... baki preprocessing...

# 2. Preprocessing - Kaggle dataset ke hisab se
df = df.drop('Loan_ID', axis=1)

# Null values fill kar
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Categorical ko numbers me convert kar
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 0, 'Not Graduate': 1})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# 3. X aur y alag karo
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 4. Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Model train karo
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Save karo model aur scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model aur Scaler save ho gaye! Accuracy:", model.score(scaler.transform(X_test), y_test))
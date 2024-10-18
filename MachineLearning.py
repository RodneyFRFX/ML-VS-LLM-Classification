import pandas as pd
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# Load the previously saved merged dataframe
merged_df = pd.read_csv('/Users/rodneyfrazier/Desktop/flight/Merged_Flight_Data.csv')

#Exploritory data Analysis

print(merged_df.head())

# Data preparation

# Define a function to classify problems related to engine
def is_engine_issue(problem_description):
    if isinstance(problem_description, str):
        if 'engine' in problem_description.lower():
            return 1
    return 0


# Apply this function to create a binary target column
merged_df['engine_issue'] = merged_df['PROBLEM'].apply(is_engine_issue)

# Drop rows with missing values for simplicity
cleaned_df = merged_df.dropna(subset=['AF TOTAL', 'TYPE', 'engine_issue'])

# Select features (we'll use aircraft type, total flight hours, engine data)
features = ['AF TOTAL', 'TYPE', 'LENG-TT', 'LENG-TSO']

# Encoding categorical variables (e.g., aircraft type) into numerical form
le = LabelEncoder()
cleaned_df['TYPE'] = le.fit_transform(cleaned_df['TYPE'].astype(str))

# Prepare X (features) and y (target)
X = cleaned_df[features].fillna(0)  # Fill missing values with 0
y = cleaned_df['engine_issue']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, target_names=['No Engine Issue', 'Engine Issue'])

print("Random Forest Classification")
print(report)


# Now, let's create a scatter plot to explore the data for outliers or anomalies

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 0]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 0]['LENG-TT'], 
            label='No Engine Issue', alpha=0.6, c='blue')
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 1]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 1]['LENG-TT'], 
            label='Engine Issue', alpha=0.6, c='red')
plt.title('Random Forest: Scatter Plot of Total Flight Hours vs Time Since Overhaul')
plt.xlabel('Total Flight Hours (AF TOTAL)')
plt.ylabel('Time Since Overhaul (LENG-TT)')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Classification")
print(classification_report(y_test, y_pred_logreg, target_names=['No Engine Issue', 'Engine Issue']))

print()
print()


plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 0]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 0]['LENG-TT'], 
            label='No Engine Issue', alpha=0.6, c='blue')
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 1]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 1]['LENG-TT'], 
            label='Engine Issue', alpha=0.6, c='red')
plt.title('Logistic Regression: Scatter Plot of Total Flight Hours vs Time Since Overhaul')
plt.xlabel('Total Flight Hours (AF TOTAL)')
plt.ylabel('Time Since Overhaul (LENG-TT)')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.svm import SVC

svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Support Vector Machines Classification")
print(classification_report(y_test, y_pred_svm, target_names=['No Engine Issue', 'Engine Issue']))


print()
print()

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 0]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 0]['LENG-TT'], 
            label='No Engine Issue', alpha=0.6, c='blue')
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 1]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 1]['LENG-TT'], 
            label='Engine Issue', alpha=0.6, c='red')
plt.title('SVM: Scatter Plot of Total Flight Hours vs Time Since Overhaul')
plt.xlabel('Total Flight Hours (AF TOTAL)')
plt.ylabel('Time Since Overhaul (LENG-TT)')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("K-Nearest Neighbors Classification")
print(classification_report(y_test, y_pred_knn, target_names=['No Engine Issue', 'Engine Issue']))

print()
print()

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 0]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 0]['LENG-TT'], 
            label='No Engine Issue', alpha=0.6, c='blue')
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 1]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 1]['LENG-TT'], 
            label='Engine Issue', alpha=0.6, c='red')
plt.title('KNN: Scatter Plot of Total Flight Hours vs Time Since Overhaul')
plt.xlabel('Total Flight Hours (AF TOTAL)')
plt.ylabel('Time Since Overhaul (LENG-TT)')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Classification")
print(classification_report(y_test, y_pred_dt, target_names=['No Engine Issue', 'Engine Issue']))

plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 0]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 0]['LENG-TT'], 
            label='No Engine Issue', alpha=0.6, c='blue')
plt.scatter(cleaned_df[cleaned_df['engine_issue'] == 1]['AF TOTAL'],
            cleaned_df[cleaned_df['engine_issue'] == 1]['LENG-TT'], 
            label='Engine Issue', alpha=0.6, c='red')
plt.title('Decision Tree: Scatter Plot of Total Flight Hours vs Time Since Overhaul')
plt.xlabel('Total Flight Hours (AF TOTAL)')
plt.ylabel('Time Since Overhaul (LENG-TT)')
plt.legend()
plt.grid(True)
plt.show()
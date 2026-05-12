# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
# from sklearn.ensemble import RandomForestClassifier

# import joblib
# #data load
# df=pd.read_csv("loan_data.csv")
# #target fixed
# y=df["not.fully.paid"]
# #input features
# X=df.drop("not.fully.paid",axis=1) 
# #converted to binary
# X=pd.get_dummies(X,drop_first=True) 
# #split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# # model selection
# model=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=10,random_state=42)

# # fit the model
# model.fit(X_train,y_train)

# #Evaluation of model
# y_prob = model.predict_proba(X_test)[:,1]
# y_pred = (y_prob > 0.3).astype(int)
# roc = roc_auc_score(y_test,y_prob)

# print(f"\n==========RANDOM FOREST==========")
# print("ROC AUC:", roc_auc_score(y_test,y_prob))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
   
# #save model
# joblib.dump(model,"loan_model.pkl")
# joblib.dump(X.columns,"columns.pkl")
# print("\nMODEL SAVED SUCCESSFULLY ✔")
import pandas as pd
df = pd.read_csv("loan_data.csv")
print(df.columns.tolist())
print("\npurpose values:", df["purpose"].unique().tolist())
print("\nlog.annual.inc sample:", df["log.annual.inc"].head(10).tolist())
print("\nint.rate sample:", df["int.rate"].head(5).tolist())
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = sns.load_dataset("iris")

X = iris.drop("species",axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

rf = RandomForestClassifier(n_estimators=10,max_depth=5)

rf.fit(X_train,y_train)

print(f"Train accuracy: {rf.score(X_train,y_train)}")
print(f"Test accuracy: {rf.score(X_test,y_test)}")

with open("model.pkl","wb") as f:
    pickle.dump(rf,f)
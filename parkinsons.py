import pandas as pd
import graphviz 
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def classify_svm(X, y, X_train, X_test, y_train, y_test):														# Accuracy: 86.44%
	clf = svm.SVC(C=100, gamma=0.01, kernel="linear")
	scores = cross_val_score(clf, X, y, cv=10)														
	print("Accuracy of Support Vector Machine: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	# clf.fit(X_train, y_train)
	# y_pred = clf.predict(X_test)
	# print(confusion_matrix(y_test, y_pred))
	# print(accuracy_score(y_test, y_pred))
	return

def classify_logistic(X_train, X_test, y_train, y_test):												# Accuracy: 84.75%
	clf = LogisticRegressionCV(cv=10, penalty="l2", random_state=42, solver='newton-cg').fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(confusion_matrix(y_test, y_pred))
	print("Accuracy of Logistic Regression: %0.2f " % accuracy_score(y_test, y_pred))
	return

def classify_tree(X, y, X_train, X_test, y_train, y_test):												# Accuracy: 80% (+/- 0.25)
	clf = tree.DecisionTreeClassifier(random_state=42)
	clf.fit(X_train, y_train)																			# Accuracy: 86.44% with no cross_val
	# scores = cross_val_score(clf, X, y, cv=10)
	# print("Accuracy of Decision Tree Classifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	y_pred = clf.predict(X_test)
	print(confusion_matrix(y_test, y_pred))
	print("Accuracy of Decision Tree Classifier: %0.2f " % accuracy_score(y_test, y_pred))
	dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True) 
	graph = graphviz.Source(dot_data) 
	graph.render("parkinsons_tree") 
	return

def grid(X_train, X_test, y_train, y_test):
	tuned_parameters = [{'kernel': ['rbf','linear','sigmoid'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
	clfv = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)	
	clfv.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print()
	print(clfv.best_params_)
	return

def data_preprocessing(X):
	# scaler = MinMaxScaler() 
	# rescaledX = scaler.fit_transform(X) 

	scaler = StandardScaler().fit(X) 
	rescaledX = scaler.transform(X) 
	return rescaledX

def do_pca(X):
	pca = PCA(n_components = 4)
	pca.fit(X)
	pca_X = pca.transform(X)
	print("Explained Variance Ratio:\t", pca.explained_variance_ratio_)  
	print("Singular Values:\t", pca.singular_values_)
	return pca_X

def main():
	data = pd.read_csv("C:/Users/DELL/Python Projects/ML mini project/parkinsons_dataset.csv")
	X = data.drop(columns={"name","status"})
	y = data.iloc[:,17]
	print("Data: \n", X.head())
	print("Target: \n", y.head())
	X = data_preprocessing(X)
	X = do_pca(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	# grid(X_train, X_test, y_train, y_test)
	classify_svm(X, y, X_train, X_test, y_train, y_test)
	classify_logistic(X_train, X_test, y_train, y_test)
	classify_tree(X, y, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
	main()
# C:\Users\DELL\Python Projects\ML mini project

# OUTPUT:

# WITH NO DATA PREPROCESSING
# Data:
#     MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)    ...      spread1   spread2        D2       PPE
# 0      119.992       157.302        74.997         0.00784    ...    -4.813031  0.266482  2.301442  0.284654
# 1      122.400       148.650       113.819         0.00968    ...    -4.075192  0.335590  2.486855  0.368674
# 2      116.682       131.111       111.555         0.01050    ...    -4.443179  0.311173  2.342259  0.332634
# 3      116.676       137.871       111.366         0.00997    ...    -4.117501  0.334147  2.405554  0.368975
# 4      116.014       141.781       110.655         0.01284    ...    -3.747787  0.234513  2.332180  0.410335

# [5 rows x 22 columns]
# Target:
#  0    1
# 1    1
# 2    1
# 3    1
# 4    1
# Name: status, dtype: int64
# Accuracy of Support Vector Machine: 0.79 (+/- 0.18)
# [[ 9  6]
#  [ 3 41]]
# Accuracy of Logistic Regression: 0.85
# [[12  3]
#  [ 5 39]]
# Accuracy of Decision Tree Classifier: 0.86



# WITH MINMAXSCALER()
# Data:
#     MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)    ...      spread1   spread2        D2       PPE
# 0      119.992       157.302        74.997         0.00784    ...    -4.813031  0.266482  2.301442  0.284654
# 1      122.400       148.650       113.819         0.00968    ...    -4.075192  0.335590  2.486855  0.368674
# 2      116.682       131.111       111.555         0.01050    ...    -4.443179  0.311173  2.342259  0.332634
# 3      116.676       137.871       111.366         0.00997    ...    -4.117501  0.334147  2.405554  0.368975
# 4      116.014       141.781       110.655         0.01284    ...    -3.747787  0.234513  2.332180  0.410335

# [5 rows x 22 columns]
# Target:
#  0    1
# 1    1
# 2    1
# 3    1
# 4    1
# Name: status, dtype: int64
# Accuracy of Support Vector Machine: 0.83 (+/- 0.24)
# [[ 9  6]
#  [ 2 42]]
# Accuracy of Logistic Regression: 0.86
# [[12  3]
#  [ 5 39]]
# Accuracy of Decision Tree Classifier: 0.86



#WITH STANDARDSCALER()
# Data:
#     MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)    ...      spread1   spread2        D2       PPE
# 0      119.992       157.302        74.997         0.00784    ...    -4.813031  0.266482  2.301442  0.284654
# 1      122.400       148.650       113.819         0.00968    ...    -4.075192  0.335590  2.486855  0.368674
# 2      116.682       131.111       111.555         0.01050    ...    -4.443179  0.311173  2.342259  0.332634
# 3      116.676       137.871       111.366         0.00997    ...    -4.117501  0.334147  2.405554  0.368975
# 4      116.014       141.781       110.655         0.01284    ...    -3.747787  0.234513  2.332180  0.410335

# [5 rows x 22 columns]
# Target:
#  0    1
# 1    1
# 2    1
# 3    1
# 4    1
# Name: status, dtype: int64
# Accuracy of Support Vector Machine: 0.84 (+/- 0.21)
# [[ 9  6]
#  [ 1 43]]
# Accuracy of Logistic Regression: 0.88
# [[12  3]
#  [ 5 39]]
# Accuracy of Decision Tree Classifier: 0.86


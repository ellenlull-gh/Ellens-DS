###  Grid Search
###  Ellen Lull
###  Machine Learning 2
###
#
### Import Libraries
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mt

import itertools

import warnings
warnings.filterwarnings("ignore")

outF = open("D:\School Stuff\DS\MLII\GridsrchOutFile.txt", "w")
outlines=["The following is displayed: List of Models, All parameters compared, Best Models \n \n"]
outF.writelines(outlines)
### Define Grid Search Main function
#
def gridfunc(X,y,clf):
    
    ret={}

###  clf is a nested dictionary.   First loop through the clf (such as KneighborsClassifier or RandomForestClassifier)
###  then for every clf found, we will loop through the parameters for the algorithm
   
    outlines = ["Models Compared \n"] 
    outF.writelines(outlines) 

    for key in clf:
        
        yhat = np.zeros(y.shape)
        cv = StratifiedKFold(n_splits=10)
        
        model_type = key
        ret[model_type]={}

        outlines = ["Classifier is: ", model_type, " \n"] 
        outF.writelines(outlines) 

        print("model type:" , model_type)
        parameter_items = clf[key].values()
        
# create all combinations of the hyperparameters for the classifier
        param_grid = list(itertools.product(*parameter_items))

# loop through all of the hyperparameters for the classifier
        for grid_combo in param_grid:
        
            idx = 0
            sklearn_call = model_type + "("

# Loop through the string of hyperparameters for each classifier and buld the string to call the classifier

            for param in clf[model_type].keys():
                
            
            #ret[model_type] = {} 
                param_value = grid_combo[idx]
                if isinstance(param_value, str):
                    sklearn_call = sklearn_call + param + " = '" + param_value + "' "
                    
                else:
                    sklearn_call = sklearn_call + param + " = " + str(param_value) 
                
                idx += 1
                             
#  if its the last parameter for the classifer fucntion, close with a parenthesis, otherwise, add a comma and continue
                    
                if idx == len(clf[model_type].keys()):
                    sklearn_call = sklearn_call + ") "
                else: 
                    sklearn_call = sklearn_call + ","
                    
# call the sklearn classifier function
                         
            classifier = eval(sklearn_call)
                      
            for train, test in cv.split(X,y):
                classifier.fit(X[train],y[train])
                yhat[test] = classifier.predict(X[test])

            total_accuracy = mt.accuracy_score(y, yhat)

            precision = mt.precision_score(y, yhat, average='micro')
            recall_score = mt.recall_score(y, yhat, average='micro')
            f1score = mt.f1_score(y, yhat, average='weighted')

            ret[model_type][sklearn_call] ={"Accuracy": total_accuracy, "F1": f1score, "Recall": recall_score, "Precision": precision}

    return(ret)

                                      
#
###  Import IRIS datasets
#
iris = datasets.load_iris()
ir = iris["data"]

X = iris["data"]
y = iris["target"]

#
### Set up Parameter Grid for searches
#

clf = {"KNeighborsClassifier": {'n_neighbors': [1,2,3,4,5,6,7,8,9,10], 'weights': ['uniform', 'distance']},
      "RandomForestClassifier": {'n_estimators': [1,2,3,4,5,6,7,8,9,10], 'criterion': ['gini', 'entropy']},
      "DecisionTreeClassifier": {'criterion': ['entropy','gini'], 'splitter': ['best','random'], 'max_depth': [1,2,3,4,5,6,7,8],'min_samples_split': [2,3,4,5],'min_samples_leaf': [1,2,3,4,5]},
      "MLPClassifier": {"hidden_layer_sizes" : [100,200], "activation": ['identity', 'logistic', 'tanh', 'relu'], "solver" : ['lbfgs', 'sgd', 'adam']}}
#
###  Call Grid Search
#
## Call the Grid Search function.   THen parse through the best models that are returned for each classifier

## Call the Grid Search function.   THen parse through the best models that are returned for each classifier

x = gridfunc(iris["data"],iris["target"],clf)
bestmdla={}
bestmdlf1={}

## Create dictionary of best model based on F1 Score and Accuracy for use in plots

for model_type in x.keys():
    print(model_type)
    b_accuracy = 0.0
    b_f1 = 0.0
    
    for parm_call, score_items in x[model_type].items():
 
        for score_type, score_value in x[model_type][parm_call].items():

            
            if score_type == "Accuracy":
 
                if score_value >  b_accuracy:
                    best_modela = parm_call
                    b_accuracy = score_value
                
            if score_type == "F1":
 
                if score_value >  b_f1:
                    best_modelf1 = parm_call
                    b_f1 = score_value
                
    bestmdla[best_modela]=b_accuracy
    bestmdlf1[best_modelf1]=b_f1
            

    outlines=["\n", "\n"]
    outF.writelines(outlines)

for bestmod, bestscore in bestmdla.items():
    outlines=["BEST MODEL ACCURACY: " , bestmod, " SCORE: ",  str(bestscore), "\n"]
    outF.writelines(outlines)
    print("BEST MODEL ACCURACY: " , bestmod, " SCORE: ",  str(bestscore))
            

for bestmod, bestscore in bestmdlf1.items():
    outlines=["BEST MODEL F1: " , bestmod, " SCORE: ",  str(bestscore), "\n"]
    outF.writelines(outlines)
    print("BEST MODEL F1: " , bestmod, " SCORE: ",  str(bestscore))
#
### Plot the best models 
### Bar Plot by best Accuracy
#
plt.figure(figsize=(10,5)) 
g=sns.barplot(x=list(bestmdla.keys()), y=list(bestmdla.values()),hue=list(bestmdla.keys()))

# create barplot put model name in legend by color

plt.title('Model Accuracy')
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.6))
plt.tick_params(axis='x', colors='w')
plt.tight_layout()

plt.savefig('D:\School Stuff\DS\MLII\modelcompare1.pdf',format='pdf')
plt.show() 


# Line graph shows more detail

model = list(bestmdla.keys())
score = list(bestmdla.values())
newmdlname=[]
for i in model:
    newmdlname.append(i[0:i.find("(")])
 

sns.lineplot(x=newmdlname, y=score)
plt.title('Model Accuracy')
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=-80)
plt.tight_layout()

plt.savefig('D:\School Stuff\DS\MLII\modelcompare2.pdf',format='pdf')
plt.show()

## Plot by F1 scores
### Bar Plot by best F1
#
plt.figure(figsize=(10,5)) 
g=sns.barplot(x=list(bestmdlf1.keys()), y=list(bestmdlf1.values()),hue=list(bestmdlf1.keys()))

# create barplot put model name in legend by color

plt.title('Model F1 Scores')
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.6))
plt.tick_params(axis='x', colors='w')
plt.tight_layout()

plt.savefig('D:\School Stuff\DS\MLII\modelcompare3.pdf',format='pdf')
plt.show() 


# Line graph shows more detail   F1 Scores

model = list(bestmdlf1.keys())
score = list(bestmdlf1.values())
newmdlname=[]
for i in model:
    newmdlname.append(i[0:i.find("(")])
 

sns.lineplot(x=newmdlname, y=score)
plt.title('Model F1 Scores')
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=-80)
plt.tight_layout()

plt.savefig('D:\School Stuff\DS\MLII\modelcompare4.pdf',format='pdf')
plt.show()



# Close out log file
outF.close()
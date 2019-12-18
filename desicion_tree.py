import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , _tree
from sklearn.utils import shuffle


fileName = "house-votes-84.data.csv"
headers = ["output"]
for i in range(1, 17):
    headers.append('col_'+str(i))

data = pd.read_csv(fileName , names= headers)

def EncodeOutData(data): 
    for col in headers[0:1]:
        data[col] = data[col].replace(to_replace = 'democrat' ,  value = 1)
        data[col] = data[col].replace(to_replace = 'republican' ,  value = 0)
    return data

def EncodeInData(data):
    for col in headers[1:]:
        data[col] = data[col].replace(to_replace = 'y' ,  value = 1)
        data[col] = data[col].replace(to_replace = 'n' ,  value = 0)
    return data

def cleanData(data): 
    for col in headers[1:]:
        yes_data =  (data[col] == 'y').sum() 
        no_data =  (data[col] == 'n').sum()
        #unknown_data = (data[col] == '?').sum()
        if yes_data >= no_data:
            data[col] = data[col].replace(to_replace = '?' ,  value ='y')
        else:
            data[col] = data[col].replace(to_replace = '?' ,  value ='n')
    return data




def calc_accuracies(data , testSize = .30):
    data = shuffle(data)
    x , y = data[headers[1:]].copy(), data[headers[0:1]].copy()
    X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size = testSize)
    classifier = DecisionTreeClassifier( )
    classifier.fit(X_train,Y_train)
    Y_predicted = classifier.predict(X_test)
    acc = round( accuracy_score(Y_test , Y_predicted)*100 , 2)
    node_count = len([x for x in classifier.tree_.feature if x != _tree.TREE_UNDEFINED])
    return acc , node_count



data = cleanData(data)
data = EncodeInData(data)
data = EncodeOutData(data)
test_size_list = [.3,.4,.5,.6,.7]
traing_size_list = [round(1.0 - i, 1) for i in test_size_list]
acc_list = []
n_nodes_list = []
for testSize in test_size_list:
        curr_acc_list = []
        for _ in range(5):   
            acc, n_nodes = calc_accuracies(data , testSize)
            curr_acc_list.append(acc)
        avg_acc = round(sum(curr_acc_list)/len(curr_acc_list) , 2)
        acc_list.append(avg_acc)
        n_nodes_list.append(n_nodes)
        
print(traing_size_list)
print(acc_list)
print(n_nodes_list)


#graph 1
plt.plot(traing_size_list , acc_list)
plt.xlabel('training')
plt.ylabel('accuracy')
plt.ylim(70 , 100 , 10)
plt.xlim(.3 , .7 , .1)
plt.show()


#graph 2
plt.plot(traing_size_list , n_nodes_list)
plt.xlabel('training')
plt.ylabel('# of nodes')
plt.ylim(1 , 50 , 1)
plt.xlim(.3 , .7 , .1)
plt.show()

import numpy as np

class KNearestNeighbor:
    def __init__(self, k):
        self.k = k

    def train(self, X, y,p_value,kind):
        self.X_train = X
        self.y_train = y
        self.p_value = p_value
        self.kind = kind
        
        return self.compute_distance_test(X) # compute_distance dan compute_distance_test yaptım
    
    def compute_distance(self, dataset):
        """
        Inefficient naive implementation, use only
        as a way of understanding what kNN is doing
        """

        num_feature = dataset.shape[1]
        num_data_row = dataset.shape[0]
        #num_train = self.X_train.shape[0]
        distances = np.zeros((num_data_row, num_data_row))

        for i in range(num_data_row):
            first_ = dataset[i]
    
            for j in range(num_data_row):
                second_= dataset[j]
                summ = 0
                for k in range(num_feature):
                    summ += (abs(first_[k]-second_[k]))**self.p_value
                    
                distances[i, j] = (summ)**(1/self.p_value)
                
                # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
                #distances[i, j] = np.sqrt( self.eps + np.sum((X_test[i, :] - self.X_train[j, :]) ** 2) )
        return distances
    
    def predict(self, X_test):
        test_data = X_test
        distances = self.compute_distance_test(test_data)
        return distances
    
    def compute_distance_test(self, dataset):
        """
        Inefficient naive implementation, use only
        as a way of understanding what kNN is doing
        """

        num_feature = dataset.shape[1] # self.X.shape[1] de kullanılabilir.
        num_data_row = dataset.shape[0]
        num_data_row_trainX = self.X_train.shape[0]
        #num_train = self.X_train.shape[0]
        distances = np.zeros((num_data_row, num_data_row_trainX))

        for i in range(num_data_row):
            first_ = dataset[i]
    
            for j in range(num_data_row_trainX):
                second_= self.X_train[j]
                summ = 0
                for k in range(num_feature):
                    summ += (abs(first_[k]-second_[k]))**self.p_value
                    
                distances[i, j] = (summ)**(1/self.p_value)
                
                # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
                #distances[i, j] = np.sqrt( self.eps + np.sum((X_test[i, :] - self.X_train[j, :]) ** 2) )
                
        results = self.predict_labels(distances)
        return results


    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[: self.k]].astype(int)
            
            if self.kind == "classification":
                unique, counts = np.unique(k_closest_classes, return_counts=True)
                outputs_numbers = dict(zip(unique, counts))
                y_pred[i] =  max(k_closest_classes, key=outputs_numbers.get) #max(k_closest_classes, key= lambda x: k_closest_classes[x]) 
                #uniques = np.unique[self.y_train]
                #y_pred[i] = np.argmax(np.bincount(k_closest_classes))
            
            if self.kind == "regression":
                y_pred[i] = sum(k_closest_classes)/len(k_closest_classes)

        return y_pred









#%%
if __name__ == "__main__":
    #*****************************
    import pandas as pd
    diabetes_data = pd.read_csv('diabetes.csv')
    
# =============================================================================
#     diabetes_data_copy = diabetes_data.copy(deep = True)
#     diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#     print(diabetes_data_copy.isnull().sum())
#     diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
#     diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
#     diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
#     diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
#     diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
#     
#     from sklearn.preprocessing import StandardScaler
#     sc_X = StandardScaler()
#     X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
#             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#            'BMI', 'DiabetesPedigreeFunction', 'Age'])
#         
#     X = diabetes_data.drop("Outcome",axis = 1)
#     y = diabetes_data_copy.Outcome
# 
# =============================================================================
    yy = diabetes_data.Outcome
    XX = diabetes_data.drop("Outcome",axis = 1)
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(XX,yy,test_size=1/3,random_state=42, stratify=yy)
    #%%
    
    KNN = KNearestNeighbor(k=3)
    deneme = KNN.train(X_train.to_numpy(), y_train.to_numpy(), p_value=3,kind="classification")
    print(f"Accuracy: {sum(deneme == y_train.to_numpy()) / y_train.shape[0]}")
    y_pred = KNN.predict(X_test.to_numpy())
    print(f"Accuracy: {sum(y_pred == y_test.to_numpy()) / y_test.shape[0]}")
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=7)
    print(knn.fit(X_train,y_train))
    print(knn.score(X_test,y_test))
    
    neighbors = np.arange(1,9)
    train_accuracy =np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i,k in enumerate(neighbors):
        #Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        
        #Fit the model
        knn.fit(X_train, y_train)
        
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
        
        #Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, y_test) 
        
    print(train_accuracy,test_accuracy) # her bir neighbor değeri için bir değer bastırıyor
        #*******************************
    #XX = np.loadtxt("diabetes.csv", delimiter=",")
    #yy = np.loadtxt("example_data/targets.txt")

    #X = np.array([[1, 1], [3, 1], [2, 3], [4, 5], [6, 6], [7, 1]])
    #y = np.array([0, 0, 0, 1, 1, 1])

    #X_test = np.array([[5, 5], [3, 3], [11, 4], [2, 3],  [1, 5]])
   
    #y_test = np.array([1, 0, 1, 0, 0])
    
    #KNN = KNearestNeighbor(k=3)
    #deneme = KNN.train(X, y, p_value=3,kind="classification")
    #y_pred = KNN.predict(X_test)
    #print(f"Accuracy: {sum(y_pred == y_test) / y_test.shape[0]}")
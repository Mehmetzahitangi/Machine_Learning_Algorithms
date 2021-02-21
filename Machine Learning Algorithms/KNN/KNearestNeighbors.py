import numpy as np

class KNearestNeighbor:
    def __init__(self, k):
        self.k = k

    def train(self, X, y,p_value,kind):
        self.X_train = X
        self.y_train = y
        self.p_value = p_value
        self.kind = kind
        
        return self.compute_distance(X)
    
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

        num_feature = dataset.shape[1] 
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
                y_pred[i] =  max(k_closest_classes, key=outputs_numbers.get)

            
            if self.kind == "regression":
                y_pred[i] = sum(k_closest_classes)/len(k_closest_classes)

        return y_pred










if __name__ == "__main__":
    X = np.array([[1, 1], [3, 1], [2, 3], [4, 5], [6, 6], [7, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[5, 5], [3, 3], [11, 4], [2, 3],  [1, 5]])
   
    y_test = np.array([1, 0, 1, 0, 0])
    
    KNN = KNearestNeighbor(k=3)
    deneme = KNN.train(X, y, p_value=3,kind="classification")
    y_pred = KNN.predict(X_test)
    print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")

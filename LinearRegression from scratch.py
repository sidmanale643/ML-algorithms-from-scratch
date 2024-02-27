import numpy as np #import the numpy library as np , used for manipulation of arrays and matrices 
                   
class LinearRegression: #create a class called linear regression
    
    #constructor that automatically initializes number of iterations and learning when the class object is created
    def __init__( self , n_iters , learning_rate) -> None: 
        
        self.n_iters = n_iters #number of iterations
        self.lr = learning_rate #learning rate 
        self.w = None 
        self.b = None 
        
    #fit the model to the training data   
    def fit(self , X , y):
        
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features) #initialize the weights as 0 with the same dimesnion as number of features
        self.b = 0  #initialize the bias as 0 
        
        #gradient descent 
        for _ in range(self.n_iters): #number of convergence steps is equal to the specified number of iterations 
        
            y_pred = np.dot(X , self.w) + self.b #hypothetical function of Linear Regression i.e. y = X * w + b
              
            grad_w = (1/num_samples) * np.dot(X.T , y_pred - y) #gradient equation of the weights with respect to the error (predicted_value - actual_value)  
            grad_b = (1/num_samples) * np.sum(y_pred - y) #gradient equation of the bias with respect to the error (predicted_value - actual_value)  

           
            self.w -= self.lr * grad_w #update the weight parameters according to the gradient 
            self.b -= self.lr * grad_b #update the bias term with according to the bias
       
        return self
   
    def predict(self, X): 
        return np.dot(X , self.w) + self.b

LR = LinearRegression(100, 0.001)
x_train = np.array([[5 , 6 , 7, 8]])  # Reshape your input to have the correct shape
y_train = np.array([4, 5, 6, 7])
LR.fit(x_train, y_train)
prediction = LR.predict(np.array([[5]]))  # Reshape your input to have the correct shape
print(prediction)



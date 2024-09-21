import numpy as np

def relu(z):
    return np.maximum(0,z)

def create_layer(n_out , n_in ):
    return np.random.randn(n_out, n_in) 
    
def forward_pass(layer , inputs):
    return relu(np.dot(layer , inputs))

x1 , x2 , x3 = 22 , 69 , 52
inputs = np.array([[x1] , [x2] ,[x3]])
      
def create_neural_net(inputs):
    
    l1 = create_layer(5 , len(inputs))
    z1 = forward_pass(l1 , inputs)
       
    l2 = create_layer(3 , 5)
    z2 = forward_pass(l2 , z1)
    
    l3 = create_layer(1 , 3)
    z3 = forward_pass(l3 , z2)   
    
    return z3


       

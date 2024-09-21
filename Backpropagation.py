import numpy as np
# loss_fn = np.mean((Z - target)**2)

# dLoss_dZ = 2*(Z - target)

# Z = z1 + z2
#activation = RELU
    #if z > 0 : then z
    #else : then 0

# dZ_dz1 = 1
# dZ_dz2 = 1

# #ignoring biases for now

# z1 = w11*x1 + w12*x2 
# dz1_dw11 = x1
# dz2_dw12 = x2

# z2 = w21*x1 + w22*x2  
# dz2_dw21 = x1
# dz2_dw22 = x2

# w11 = w11 - lr * grad(w11)
# w12 = w12 - lr * grad(w12)
# w21 = w21 - lr * grad(w21)
# w22 = w22 - lr * grad(w22)


def activate(z):
    if z > 0:
        return z
    else:
        return 0  

def forward_pass(x1 , x2 , target , w11 , w12 , w21 , w22 ):
    z1 = w11*x1 + w12 * x2
    z1 = activate(z1) + 0.001
    
    z2 = w21*x1 + w22 * x2
    z2 = activate(z2) + 0.001
    
    z = z1 + z2
    
    loss = np.mean((z - target)**2)
    return loss , z


def backward_pass(x1 , x2 , target , w11 , w12 , w21 , w22 , z , lr):
    
    #derivative of loss function (loss_function = ((predicted-target)**2)_n) with respect to activation of last Layer
    dLoss_dz = 2*(z - target)
    
    #z = z1 + z2 since there are 2 neurons in the previous Layer
    dz_dz1 = 1
    dz_dz2 = 1
    
    #z1 = w11 * x1 + w12 * x2
    dz1_dw11 = x1
    dz1_dw12 = x2
    
    #z2 = w21 *x1 + w22 * x2
    dz2_dw21 = x1
    dz2_dw22 = x2
    
    #applying chain rule

    dLoss_dw11 = dLoss_dz * dz_dz1 * dz1_dw11
    w11_up = w11 - lr * dLoss_dw11
    
    dLoss_dw12 = dLoss_dz * dz_dz1 * dz1_dw12
    w12_up = w12 - lr * dLoss_dw12

    dLoss_dw21 = dLoss_dz * dz_dz2 * dz2_dw21
    w21_up = w21 - lr * dLoss_dw21

    dLoss_dw22 = dLoss_dz * dz_dz2 * dz2_dw22
    w22_up = w22 - lr * dLoss_dw22
    
    return w11_up, w12_up, w21_up, w22_up


def fit(x1 , x2 , target , lr , epochs):
    
    w11 = np.random.randn()
    w12 = np.random.randn()
    w21 = np.random.randn()
    w22 = np.random.randn()
    for epoch in range(epochs):
        loss , z = forward_pass(x1 , x2 , target , w11 , w12 , w21 , w22 )
        print(f"For Epoch: {epoch} Loss:{loss}")
        w11, w12, w21, w22 = backward_pass(x1 , x2 , target , w11 , w12 , w21 , w22 , z , lr)

fit(11 , 29 , 69, 0.0001 , 15)

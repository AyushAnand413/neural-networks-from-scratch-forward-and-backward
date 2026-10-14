"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    
    #the gardients which we will be calculating later needs to be verifed that calc was correct or wrong so we need 
    #as one small calc mistake can affect the whole neural network
    #so via using central diff concept we will estimate grads
    #f'(x) = (f(x+e) - f(x-e))/2e
    #so without using derivatiove we are estimating the derivative 

    grads = np.zeros_like(x, dtype=float) #array of zeroes to fill calc grads

    #now for each element we need to cal 
    for idx in np.ndindex(x.shape):
        #ndindex means n dimensional aaray
        org = x[idx]
        x[idx] = org + eps
        f_plus = f(x) #see we pass the changed vallue to fn and get a scalar value
        x[idx] = org-eps
        f_minus = f(x)
        
        grads[idx] = (f_plus - f_minus)/(2*eps)
        x[idx]=org 
    return grads

# Step 2 - gradient_check (not yet solved)
# TODO: implement

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement


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

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    #so here we measure the max relative error in between the gards calculated by analytic grad(maths derivative backprop) and numerical_grad(central diff)

    analytic_grad = np.asarray(analytic_grad,dtype =float)
    numeric_grad = np.asarray(numeric_grad,dtype=float)

    diff = np.abs(analytic_grad-numeric_grad)

    #scale is the larger absolute gradient at each position, with tol as the minimum value to prevent division by zero or a very small number
    scale = np.maximum(
        np.maximum(np.abs(analytic_grad),np.abs(numeric_grad)),
        tol 
        )
    
    relative_error = diff/scale

    return float(np.max(relative_error))
    #this will be used ahead to check if the grad calc are correct or not

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    #we have toi build dense layer from scratch 
    #y = wx+b
    #so we need to store params build forward layer and backward and return these
    W,b = weight_init_fn(in_dim,out_dim)
    params = {'W':W , 'b':b}

    def forward(x):
      y = x @ params['W'] + params['b'] #y = wx + b
      cache = x  #to store the org input as we need it to calc grads
      return y,cache
    
    def backward(dout,cache):
      x = cache

      dx = dout @ params['W'].T #Passed backward to the previous layer
      dW = x.T @ dout
      db = np.sum(dout,axis=0)
      grads = {'W':dW , 'b':db}
      return dx,grads
  
    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    params = {}

    def forward(x):
      #here x is output of previous dense layer 
      y= np.maximum(0,x)
      cache = x
      return y,cache

    def backward(dout,cache):
      x= cache
      dx = dout *(x>=0) # pass grads where input is non negative
      return dx,{}

    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

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


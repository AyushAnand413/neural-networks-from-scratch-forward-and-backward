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
      dx = dout *(x>0) # pass grads where input is non negative
      return dx,{}

    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    """
    The goal is to ensure that:
      - Different neurons start with different weights.
      - Weights aren't all zero.
      - Weight values are scaled appropriately so signals don't vanish or explode as they pass through layers.
    """   
    if scheme=='he':
      W = np.random.randn(in_dim,out_dim) * np.sqrt(2.0/ in_dim) 
      #he intilization for relu
    else:
       W = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / in_dim)  # Alternative fan-in scaling

    b = np.zeros(out_dim) # biases to 0

    return W,b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """

    def loss_fn(logits,y_true):

      shifted = logits - np.max(logits, axis=1 , keepdims=True)
      #we do this so exp doesnt overflow
      #now convert shifted logits to probab using softmax
      exp_logits = np.exp(shifted)
      sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
      probs = exp_logits/sum_exp

      #select logit of the correct class 
      correct_logits = logits[np.arange(len(y_true)), y_true]

      #loss
      log_sum_exp = np.log(sum_exp.ravel()) + np.max(logits, axis=1)
      loss = np.mean(log_sum_exp-correct_logits)

      #gradient of cross-entropy loss with respect to the logits
      n = len(y_true)
      d_logits = probs.copy()
      d_logits[np.arange(n), y_true] -= 1
      d_logits /= n

      return float(loss), d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    #Each layer has its own parameter dictionary.
    params = [layer["params"] for layer in layers]

    def forward(x):
      
      caches = []

      for layer in layers:
        x , cache = layer["forward"](x) #calls the current layers forward function the output becomes the input for the next layer
        caches.append(cache)

      return x,caches
    
    def backward(dout,caches):
      grad_list = [{} for _ in layers] # one empty grad dictonary for every layer

      #now here we need bottom to up
      for i in reversed(range(len(layers))):
        
        dout,grads = layers[i]["backward"](dout, caches[i])
        grad_list[i] = grads

      return dout, grad_list

    return{
      "forward" : forward,
      "backward" : backward,
      "params" : params
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    #this does one complete forward and backward pass for a batch
    logits , caches = model["forward"](x) #model run forward

    loss, d_logits = loss_fn(logits,y) # calc loss 

    dx , param_grads = model["backward"](d_logits,caches)#backprop
    return float(loss), param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    #now the grads we got via backrprop will be used to update weights
    
    if lr<=0:
      raise ValueError("learning rate must be  positive")
    if kind != 'sgd':
      raise ValueError("unsupported optimizer")

    def update(param, grad):
      #This function receives one parameter structure and its matching gradient structure.Why do we need it? Because parameters might be nested at different levels.   

      if isinstance(param,np.ndarray):
        param[...] -= lr*grad  #... means modify exsiting array

      elif isinstance(param, dict):
        for key in param:
          update(param[key],grad[key])

      elif isinstance(param,list):
        for p,g in zip(param,grad):
          update(p,g) 

    def step(grads):
      update(params,grads)

    return{
      "step":step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss,grads = forward_backward(model,loss_fn,x_batch,y_batch)
    #this call forward pass calc batch loss  and perform backprop to calc gradients

    optimizer["step"](grads) #passes the gards to optimizer

    return float(loss)

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement


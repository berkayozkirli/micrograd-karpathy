from math import exp
class Value:
    # Sinle value
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # The function that will take us backwards, will be initiliaized in an op function 
        self._backward = lambda: None
        # The variable that will be used to build the tree, will be initiliaized in an op function 
        self._children = _children
        # The operation that produced this value, will be initiliaized in an op function 
        self._op = _op
    # Basic operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "x")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out
    # Some activation functions
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += 0 if self.data < 0 else 1 * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Value((exp(2 * self.data) - 1)/(exp(2 * self.data) + 1), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out 

    def sigmoid(self, _beta = 1):
        out = Value(1/(1 + exp(-1 * _beta * self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.data * (_beta - _beta * out.data)
        out._backward = _backward

        return out
    
    def swish(self, _beta = 1):
        return self.sigmoid(_beta) * self.data 

    # This function topologically sorts the tree of operations to find gradients by going backwards
    def backward(self):
        topologically_sorted = []
        visited = set()
        def topologically_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    topologically_sort(child)
                topologically_sorted.append(node)   
        topologically_sort(self)
        # Order in which the backpropagation takes place
        backwards_order = topologically_sorted.reverse()
        # Initialize to one, else all will be zero
        self.grad = 1.0
        for node in backwards_order:
            node._backward 

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
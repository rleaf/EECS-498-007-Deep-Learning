def toads(f, x):
   k = f(x)
   print('ye')
   return k


def frogs(x, y):
   loss = x + y
   return loss

x = 4
y = 4
frogs1 = toads(lambda x: frogs(x, y), x)
print(frogs1)

# 1) print frogs1 function. Calling frogs1 doesn't require argument because the lambda is being passed as an arg
# 2) toads() takes lambda (f) and x (x) as arguments
# 3) k = (lambda x:)(x), which immediately evaluates the lambda with x as the argument
# 4) toads() returns k 


# g = lambda x: x + 2
# print(g(2))
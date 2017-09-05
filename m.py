a=[{'a':1,'b':1},{'a':2,'b':1}]
b,c=[(x['a'],x['b']) for x in a]
print(b,c)
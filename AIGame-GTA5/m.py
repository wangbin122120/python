a=[[x,'1']  for x in range(10)]
print(len(a))
print(a)
print(a[:3])
print(a[:3][:6])


def p(t=5):
    for x in a[-1*t:]:
        print(x)
p(5)
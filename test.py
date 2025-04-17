import random

def random_from_column(column):
    val = []
    count = []
    for entry in column:
        if not entry in val:
            val.append(entry)
            count.append(1)
        else:
            count[val.index(entry)] += 1
    rd = random.randint(1, len(column))
    print(count)
    print(rd)
    index = 0
    while True:
        rd -= count[index]
        if rd > 0:
            index += 1
        else:
            return val[index]
        

a = [1,1,2,2,1,1,1,1,3,3,1,1,1]
p = [0, 0, 0]
for i in range(1000000):
    p[random_from_column(a)-1]+=1
print(p)
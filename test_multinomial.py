import math
from collections import Counter


def multinomial(params):
    res = math.factorial(sum(params))
    for p in params:
        res //= math.factorial(p)
    return res


print(multinomial([3, 1, 1, 59]))

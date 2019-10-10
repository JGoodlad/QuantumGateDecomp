import math
from typing import *

import numpy as np

def get_qc_name():
    return 'Aspen-4-12Q-A'

def is_throw_set():
    return True

def throw_if_set():
    if is_throw_set():
        raise Exception("throw_if_set evaluted to true")


def suppress_exception(f):
    try:
        f()
    except:
        pass
    return

def getAll1Indexes(s: str):
    return [i for i, v in enumerate(s) if v == '1']

def getAll0Indexes(s: str):
    return [i for i, v in enumerate(s) if v == '0']

def getNegation(s: str):
    res = ''
    for i in s:
        if i == '0':
            res += '1'
        else:
            res += '0'
    return res

def getAllBitStrings(n):
    base = ['0', '1']
    result = list(base)
    for i in range(1, n):
        t = []
        for b in base:
            for j in result:
                t.append(b + j)
        result = t
    return result    

def getIdentityN(n):
    return np.identity(2 ** n)

def getToffoliSelectN(n):
    t = getIdentityN(n) #2^n x 2^n identity
    t[-1, -1] = 0
    t[-2, -2] = 0
    t[-1, -2] = 1
    t[-2, -1] = 1
    return t

def xorString(s1: str, s2: str) -> str:
    res = ''
    for b1, b2 in zip(s1, s2):
        res += '0' if b1 == b2  else '1'
    return res

def getUniqueRandomValue(n, usedValues:set = set()) -> str:
    while True:
        res = "".join(np.random.choice(['0', '1'], size = n))
        if res not in usedValues:
            usedValues.add(res)
            return res

def filterNonZeroArrays(arrays):
    res = []
    for a in arrays:
        if np.mean(a) != 0:
            res.append(a)
    return res

def extractResults(results):
    (res, i) = ([], 0)
    while True:
        if i in results:
            res.append(np.mean(results[i]))
            i += 1
        else:
            break
    return res

def extractResultsPerRunForQubits(results : dict, *args):
    numberOfRuns = len(next(iter(results.values())))
    res = []
    for i in range(numberOfRuns):
        t = []
        for q in args:
            t.append(int(results[q][i]))
        res.append(t)
    return res

def extractResultsForQubits(results : dict, *args):
    (res, i) = ([], 0)
    for i in args:
        res.append(np.mean(results[i]))
    return res

def uniqueLists(l: List[List]) -> List[List]:
    return list(map(lambda x: list(x), set(tuple(x) for x in l)))

def toBitArray(bs: str) -> List[int]:
    t = []
    for b in bs:
        t.append(int(b, 2))
    return t

def toBitString(ba):
    s = ''
    for b in ba:
        s += str(int(round(b)))
    return s
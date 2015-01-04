#!/usr/bin/env python

import sys
import subprocess

def scan(f, state, it):
  for x in it:
    state = f(state, x)
    yield state
    
def runProg(bin, stdin = None):
    pipe = subprocess.Popen([bin],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (out, err) = pipe.communicate()
    return out

def test(n):
    seq = [1] * n
    with open('input.txt', 'w') as f:   
        f.write(str(n) + "\n")
        in_test = map(lambda x: str(x), seq)
        f.write(' '.join(in_test))
    runProg(sys.argv[1])
    expect = list(scan(lambda x,y: x + y, 0, seq))
    with open('output.txt', 'r') as f:
        data = f.read()
        out = map(lambda x: float(x), data.strip().split(' '))
        test = filter(lambda (x,y): x != y, zip(out, expect))
        if len(test) == 0:
            print("OK : [1] * " + str(n))
        else:
            print("FAIL [1] * " + str(n))
            print("get : " + str(test[0][0]) + " expect : " + str(test[0][1]))    
        

def main():
    if len(sys.argv) < 2:
        print("usage : ./test.py executable_name")
        return
    if len(sys.argv) > 2:
        test(int(sys.argv[2]))
        return
    test(5)
    n = 48576;
    while(n <= 1048576):
        test(n)
        n += 100000
        
        
main()
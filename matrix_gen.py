#!/usr/bin/python

import sys
import random

for col in range(int(sys.argv[1])):
    for row in range(int(sys.argv[2])):
        if 'd' == sys.argv[3]:
            sys.stdout.write(str(1 if row == col else 0))
        else:
            sys.stdout.write(str(random.random()))    
            
        sys.stdout.write(' ')    

    sys.stdout.write('\n')
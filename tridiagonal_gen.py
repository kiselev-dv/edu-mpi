#!/usr/bin/python
import sys
import random

len = int(sys.argv[1])
for i in range(len):
    random.random()
    sys.stdout.write(str(random.random()) + ' ' + str(random.random()) + ' ' + str(random.random()) + ' ' + str(random.random()) + '\n')

#!/usr/bin/python

from mpi4py import MPI
import logging
from getopt import getopt, GetoptError
import itertools
import random
from math import sqrt
import sys

logging.basicConfig(level=logging.INFO)

def parseOpts(options):
    n = 10000
    for opt, arg in options:
        if opt in ("-n"):
            n = int(arg)
            if n <= 0:
                print '-n is too small.'
                sys.exit(1)
    
    return n

def monteCarlo(n):
    result = 0;
    for _ in itertools.repeat(None, n):
        if sqrt(random.random() ** 2 + random.random() ** 2) <= 1:
            result += 1; 

    return result; 

#Nilakantha
def nilSequence(n):
    if n == 0:
        return 3
    
    sign = 1
    if n % 2 == 0:
         sign = -1
         
    base = n * 2 
    return sign * 4.0 / (base * (base + 1) * (base + 2))

def sequenceSumm(rng):
    r = 0;
    for n in rng:
        r += nilSequence(n)
    
    return r
            
comm = MPI.COMM_WORLD
logging.debug('Got comm interaface')

rank = comm.Get_rank()
threads = comm.Get_size()
Master = rank == 0

log = logging.getLogger('Thread {0}'.format(rank))
log.debug('Initialized')

steps = 10000
try:
    opts, args = getopt(sys.argv[1:], "n:")
    steps = parseOpts(opts)
except GetoptError:
    sys.exit(1)    


local_n = comm.bcast(steps / threads, root = 0)
total = local_n * threads   

#loc_result = monteCarlo(local_n)
loc_result = sequenceSumm(range(rank * local_n, rank * local_n + local_n))
tot_result = comm.reduce(loc_result, op=MPI.SUM)

if Master:
    #pi_estimate = 4 * float(tot_result) / float(total)
    pi_estimate = tot_result
    print pi_estimate

log.debug("All done")
             

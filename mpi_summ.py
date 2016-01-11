#!/usr/bin/python

from mpi4py import MPI
import logging
import sys
from getopt import getopt, GetoptError
import numpy
import math

logging.basicConfig(level=logging.INFO)

def parseInputSource(options):
    for opt, arg in options:
        if opt in ("-s", "--sequence"):
            return range(1, int(arg) + 1).__iter__()
        elif opt in ("-f", "--file"):
            if '-' == arg:
                return sys.stdin
            
            with open(arg) as f:
                return f
            
comm = MPI.COMM_WORLD
logging.debug('Got comm interaface')

rank = comm.Get_rank()
threads = comm.Get_size()

log = logging.getLogger('Thread {0}'.format(rank))

log.debug('Initialized')

Master = rank == 0
# Master thread input read routine
if Master: 
    inpS = None
    try: 
        opts, args = getopt(sys.argv[1:], 'hs:f:', ['sequence=', 'file='])
        inpS = parseInputSource(opts)
    except GetoptError:
        print 'mpi_summ.py {-s N, -f file}'
        sys.exit(2)
        
    if inpS == None:
        log.error("Can't get input source")
        sys.exit(3)
    
    inp = [ float(str(i).rstrip('\n')) for i in inpS ]
    
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Expected: {0}".format(numpy.sum(inp)))
    
    data = [];
    for d in range(threads):
        data.append([])
    
    counter = 0;
    for f in inp:
        data[counter % threads].append(f)
        counter += 1;

else:    
    data = None

if Master:
    log.debug("Data to scatter: {0}".format(data))
    
localArray = comm.scatter(data, root = 0)

log.debug('Local Array {0}'.format(localArray))

localSumm = numpy.sum(localArray)
log.debug("Local summ: {0}".format(localSumm))

steps = int(math.log(threads, 2)) + 1
if Master:
    log.debug("Doubling steps {0}".format(steps))

for step in range(1, steps + 1):
    recivrs = range(0, threads, 2 ** step)
    senders = range(2 ** (step - 1), threads, 2 ** step)
    
    send = dict();
    for i in range(len(senders)):
        send[senders[i]] = recivrs[i]
    
    if Master:
        log.debug("Step {0}, Communications {1}".format(step, send))
    
    rciv = {v: k for k, v in send.items()}    
    if rank in send:
        comm.send(localSumm, dest=send[rank], tag=1)
    elif rank in rciv:
        localSumm += comm.recv(source=rciv[rank], tag=1)
             
if Master:
    print localSumm             

log.debug("All done")
             

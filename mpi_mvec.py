#!/usr/bin/python

from mpi4py import MPI
import logging
from getopt import getopt, GetoptError
import sys

logging.basicConfig(level=logging.DEBUG)

def parseOpts(options):
    for opt, arg in options:
        if opt in ("-m", "--matrix"):
            matrixF = open(arg)
        if opt in ("-v", "--vector"):
            vectorF = open(arg)
            
    return matrixF, vectorF

def partialSumm(row, vecArr):
    sumR = 0;
    
    for i in range(len(row)):
        sumR += row[i] * vecArr[i]
    
    return sumR;

comm = MPI.COMM_WORLD
logging.debug('Got comm interaface')

rank = comm.Get_rank()
threads = comm.Get_size()
Master = rank == 0

log = logging.getLogger('Thread {0}'.format(rank))
log.debug('Initialized')

mtx = None 
vec = None

if Master:
    try:
        opts, args = getopt(sys.argv[1:], "m:v:")
        mtx, vec = parseOpts(opts)
    except GetoptError:
        log.error("Can't parse arguments")
        sys.exit(1)
        

vecArr = None
if Master:
    vecArr = [ float(i.rstrip('\n')) for i in vec ]
    log.info('Vector: {0}'.format(vecArr))

vecArr = comm.bcast(vecArr, root = 0)

row2part = dict();        
if Master:
    c = 0
    log.info('Distribute matrix')
    for lineS in mtx:
        row = [ float(i) for i in lineS.split(' ') ]
        targTh = c % threads
        
        if targTh != 0:
            comm.send(c, dest=targTh, tag=1)
            comm.send(row, dest=targTh, tag=2)
        else:
            row2part[c] = partialSumm(row, vecArr)
        
        c += 1;
    
    for t in range(1, threads):
        comm.send(-1, dest=t, tag=1)
    
    log.info('Done matrix distribution')    
else:
    while True:
        rn = comm.recv(source = 0, tag=1)
        log.debug('Rcvd: {0}'.format(rn))
        if rn < 0:
            break
        
        row = comm.recv(source = 0, tag=2)
        log.debug('Rcvd: {0}'.format(row))
        
        row2part[rn] = partialSumm(row, vecArr)
    
    log.info('Rows accuired')

rVector = [] 
if Master:
    for t in range(1, threads):
        parts = comm.recv(source = t, tag = 3)
        for k, v in parts.items():
            rVector.insert(k, v)
    
    for k, v in row2part.items():    
        rVector.insert(k, v)
        
else:
    comm.send(row2part, dest = 0, tag = 3)

if Master:
    print rVector
             

#!/usr/bin/python

from mpi4py import MPI
import logging
from getopt import getopt, GetoptError
import sys
import numpy

logging.basicConfig(level=logging.INFO)

def parseOpts(options):
    for opt, arg in options:
        if opt in ("-a"):
            a = open(arg)
        if opt in ("-b"):
            b = open(arg)
            
    return a, b

def writePart(partials, rowsByIndex, rn, rowB):
    for bi in range(len(rowB)):
        for ai, rowA in rowsByIndex.items():
            r = rowA[rn] * rowB[bi]
            k = str(ai) + ' ' + str(bi)
            
            if r > 0:
                if not k in partials:
                    partials[k] = 0

                partials[k] = partials[k] + r
            

comm = MPI.COMM_WORLD
logging.debug('Got comm interaface')

rank = comm.Get_rank()
threads = comm.Get_size()
Master = rank == 0

log = logging.getLogger('Thread {0}'.format(rank))
log.debug('Initialized')

mtxA = None 
mtxB = None

if Master:
    try:
        opts, args = getopt(sys.argv[1:], "a:b:")
        mtxA, mtxB = parseOpts(opts)
    except GetoptError:
        log.error("Can't parse arguments")
        sys.exit(1)
        

rowsByIndex = dict();        
rowsA = 0
if Master:
    log.info('Distribute matrix A')
    for lineS in mtxA:
        try:
            row = [ float(i) for i in lineS.strip().split(' ') ]
            
            targTh = rowsA % threads
            
            if targTh != 0:
                comm.send(rowsA, dest=targTh, tag=1)
                comm.send(row, dest=targTh, tag=2)
            else:
                rowsByIndex[rowsA] = row
            
            rowsA += 1;
        except ValueError:
            log.error('Failed to parse: {0}'.format(lineS))
            sys.exit(6)
    
    for t in range(1, threads):
        comm.send(-1, dest=t, tag=1)
    
    log.info('Done matrixA distribution')    
else:
    while True:
        rn = comm.recv(source = 0, tag=1)
        log.debug('Rcvd: {0}'.format(rn))
        if rn < 0:
            break
        
        row = comm.recv(source = 0, tag=2)
        log.debug('Rcvd: {0}'.format(row))
        
        rowsByIndex[rn] = row
    
    log.debug('Rows accuired')

# Matrix A distributed
# Distribute Matrix B

partials = dict()
colsB = 0
if Master:
    c = 0
    log.info('Distribute matrix B')
    for lineS in mtxB:
        row = [ float(i) for i in lineS.strip().split(' ') ]
        colsB = len(row)
        
        comm.bcast(c, root=0)
        row = comm.bcast(row, root=0)
        writePart(partials, rowsByIndex, c, row)
        
        c += 1
    
    for t in range(1, threads):
        comm.bcast(-1, root=0)
    
    log.info('Done matrixB distribution')    
else:
    while True:
        rn = comm.bcast(0, root=0)
        log.debug('Rcvd: {0}'.format(rn))
        if rn < 0:
            break
        
        row = comm.bcast(row, root=0)
        log.debug('Rcvd: {0}'.format(row))
        
        writePart(partials, rowsByIndex, rn, row)
    
    log.info('Rows accuired')
    

rMatrix = numpy.zeros(rowsA * colsB).reshape(rowsA, colsB)
if Master:
    for t in range(1, threads):
        parts = comm.recv(source = t, tag = 8)
        log.debug('Partials {0}'.format(parts))
        for k, v in parts.items():
            ai = k.split(' ')[0]
            bi = k.split(' ')[1]
            rMatrix[ai, bi] = rMatrix[ai, bi] + v 
    
    log.debug('Partials {0}'.format(partials))
    for k, v in partials.items():    
        ai = k.split(' ')[0]
        bi = k.split(' ')[1]
        rMatrix[ai, bi] = rMatrix[ai, bi] + v
        
else:
    comm.send(partials, dest = 0, tag = 8)
    
    
if Master:
    for row in rMatrix:
        for e in row:
            sys.stdout.write(str(e))
            sys.stdout.write(' ')
        sys.stdout.write('\n')   
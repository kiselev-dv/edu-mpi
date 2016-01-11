#!/usr/bin/python

from mpi4py import MPI
import logging
from getopt import getopt, GetoptError
import sys

logging.basicConfig(level=logging.INFO)

SPLIT_ARRAY = 1

A = 0
B = 1
C = 2
F = 3

def getAB(cond, coef):
    if coef is None:
        return ( -cond[C] / cond[B], cond[F] / cond[B] )
    
    else:
        y = cond[B] + cond[A] * coef[0]
        if y == 0:
            return None
        
        alpha = -cond[C] / y
        betha = ( cond[F] - cond[A] * coef[1] ) / y
        return (alpha, betha)

def getSN(cond, coef):
    if coef is None:
        return ( -cond[A] / cond[B], cond[F] / cond[B] )
    
    else:
        y = cond[B] + cond[C] * coef[0]
        if y == 0:
            return None
        
        xsi = -cond[A] / y
        enn = ( cond[F] - cond[C] * coef[1] ) / y
        return (xsi, enn)
    
def getX(coef, lastx):
    if lastx is None:
        return coef[1]
    
    return coef[0] * lastx + coef[1]     

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class MpiTDSolver:

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        logging.debug('Got comm interaface')

        self.rank = self.comm.Get_rank()
        self.threads = self.comm.Get_size()
        self.Master = self.rank == 0

        self.log = logging.getLogger('Thread {0}'.format(self.rank))
        self.log.debug('Initialized')
        
        self.midleX = None
        self.result = []

    def parseArgs(self):
        """ Parse arguments
                self.src          source file/stdin
                slef.right
        """
        try:
            opts, args = getopt(sys.argv[1:], "f:lr:")
        except GetoptError, msg:
            logging.error(msg)
            sys.exit(2)
        
        self.round = 0
        self.right = True    
        for opt, arg in opts:
            if opt in ("-f"):
                if '-' == arg:
                    self.src = sys.stdin
                else:    
                    self.src = open(arg)
            if opt in ("-l"):
                self.right = False        
            if opt in ("-r"):
                self.round = int(arg)         

    def readArray(self):
        
        self.rows = []
        for lineS in self.src:
            cond = [ float(i) for i in lineS.strip().split(' ') ]
            self.rows.append(cond)

        self.log.debug('Done read')       

    def splitArray(self):
        if self.Master:
            self.rows = []
            for lineS in self.src:
                self.rows.append([ float(i) for i in lineS.strip().split(' ') ])
        
        data = None
        if self.Master:
            l = len(self.rows)
            r = l % 2
            data = list(chunks(self.rows, l // 2 + r)) 
            self.log.debug('Reshaped array: {0}'.format(data))       
            
        self.rows = self.comm.scatter(data, root = 0)
        self.log.debug('Done split array: {0}'.format(self.rows))   

    def runForwardRight(self):
        self.coeff = []

        prevCond = None
        for row in self.rows:
            prevCond = getAB(row, prevCond)
            if prevCond is not None:
                self.coeff.append(prevCond)
            
        self.log.debug('Done Forward Right')        

    def runBackwrdRight(self):
        lastX = self.midleX
        for coef in reversed(self.coeff):
            lastX = getX(coef, lastX)
            self.result.append( lastX )
        
        self.result.reverse()
        self.log.debug('Done Backward Right')
        
                
    def runForwardLeft(self):
        self.coeff = []

        prevCond = None
        for row in reversed(self.rows):
            prevCond = getSN(row, prevCond)
            if prevCond is not None:
                self.coeff.append(prevCond)
            
        self.log.debug('Done Forward left')        


    def runBackwrdLeft(self):
        lastX = self.midleX
        for coef in reversed(self.coeff):
            lastX = getX(coef, lastX)
            self.result.append( lastX )
        
        self.log.debug('Done Backward Left')
            

    def mergeBidirectional(self):
        ab = None
        sn = None
        if self.Master:
            ab = self.coeff[-1]
            self.comm.send(ab, dest=1 ,tag=2)
            sn = self.comm.recv(source=1 ,tag=2)
        else:
            sn = self.coeff[-1]
            ab = self.comm.recv(source=0 ,tag=2)
            self.comm.send(sn, dest=0 ,tag=2)
        
        self.midleX = (sn[1] + sn[0] * ab[1]) / (1 - sn[0] * ab[0])
        if not self.Master:
            self.result.append(self.midleX)
            del self.coeff[-1] 
            
        
        self.log.debug('Merged X: {0}'.format(self.midleX))
    
    def mergeResults(self):
        if self.Master:
            self.result.extend(self.comm.recv(source = 1, tag = 3))
        else:
            self.comm.send(self.result, dest = 0, tag = 3)
            
    def run(self):
        self.parseArgs()
        
        if self.threads == 1:
            self.readArray()
            if self.right:
                self.runForwardRight()
                self.runBackwrdRight()
            else:    
                self.runForwardLeft()
                self.runBackwrdLeft()
            
        elif self.threads == 2:
            self.splitArray()
            
            if self.Master :
                self.runForwardRight()
            else:
                self.runForwardLeft()    
            
            self.mergeBidirectional()
            if self.Master :
                self.runBackwrdRight()
            else:
                self.runBackwrdLeft()
            
            self.mergeResults()
            
        else:
            self.log.error('Specified for one or two threads')
            sys.exit(1)    
        
        if self.Master:
            for f in self.result:
                if self.round == 0: 
                    sys.stdout.write(str(f))
                else:
                    s = str(round(f, self.round))
                    if s[0] == '-' and all( c == '0' or c == '.' for c in list(s)[1:]):
                        s = s[1:]
                    sys.stdout.write(s)    
                sys.stdout.write(' ')
            sys.stdout.write('\n')

if __name__ == "__main__":
    MpiTDSolver().run()
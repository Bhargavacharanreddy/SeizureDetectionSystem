# WalshHadamard.py
#
# A module containing functions for derivation of 1-D Walsh-Hadamard Transform
# and performing WHT Significance Test based on Oprina et al. (2009)
#
# (c) 2015 QuantAtRisk.com, written by Pawel Lachowicz

# Log of changes:
#   0.0.1   Apr 07, 2015    package
#   0.0.2   Apr 09, 2015    added functions:
#                             - bit_reverse_traverse
#                             - get_bit_reversed_list(l)
#                             - FWHT(X) (Fast Walsh-Hadamard Transform)

from math import log, trunc, sqrt
from scipy.special import gammainc
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.stats import norm, chi2 as Chi2
import numpy as np

def Hadamard2Walsh(n):
    # Function computes both Hadamard and Walsh Matrices of n=2^M order
    # (c) 2015 QuantAtRisk.com, coded by Pawel Lachowicz adopted after
    #  au.mathworks.com/help/signal/examples/discrete-walsh-hadamard-transform.html
    import numpy as np
    from scipy.linalg import hadamard
    from math import log

    hadamardMatrix=hadamard(n)
    HadIdx = np.arange(n)
    M = log(n,2)+1

    for i in HadIdx:
        s=format(i, '#032b')
        s=s[::-1]
        s=s[:-2]
        s=list(s)
        x=[int(x) for x in s]
        x=np.array(x)
        if(i==0):
            binHadIdx=x
        else:
            binHadIdx=np.vstack((binHadIdx,x))

    binSeqIdx = np.zeros((n,M)).T

    for k in reversed(xrange(1,int(M))):
        tmp=np.bitwise_xor(binHadIdx.T[k],binHadIdx.T[k-1])
        binSeqIdx[k]=tmp

    tmp=np.power(2,np.arange(M)[::-1])
    tmp=tmp.T
    SeqIdx = np.dot(binSeqIdx.T,tmp)

    j=1
    for i in SeqIdx:
        if(j==1):
            walshMatrix=hadamardMatrix[i]
        else:
            walshMatrix=np.vstack((walshMatrix,hadamardMatrix[i]))
        j+=1

    return (hadamardMatrix,walshMatrix)

def WHT(x):
    x=np.array(x)
    if(len(x.shape)<2): # make sure x is 1D array
        if(len(x)>3):   # accept x of min length of 4 elements (M=2)
            # check length of signal, adjust to 2**m
            n=len(x)
            M=trunc(log(n,2))
            x=x[0:2**M]
            h2=np.array([[1,1],[1,-1]])
            for i in xrange(M-1):
                if(i==0):
                    H=np.kron(h2,h2)
                else:
                    H=np.kron(H,h2)

            return (np.dot(H,x)/2.**M, x, M)
        else:
            print("HWT(x): Array too short!")
            raise SystemExit
    else:
        print("HWT(x): 1D array expected!")
        raise SystemExit


def line():
    print("-"*70)

def ret2bin(x):
    # Function converts list/np.ndarray values into +/-1 signal
    # (c) 2015 QuantAtRisk.com, by Pawel Lachowicz
    Y=[]; ok=False
    if('numpy' in str(type(x)) and 'ndarray' in str(type(x))):
        x=x.tolist()
        ok=True
    elif('list' in str(type(x))):
        ok=True
    if(ok):
        for y in x:
            if(y<0):
                Y.append(-1)
            else:
                Y.append(1)
        return Y
    else:
        print("Error: neither 1D list nor 1D NumPy ndarray")
        raise SystemExit


def xsequences(x):
    x=np.array(x)
    if(len(x.shape)<2): # make sure x is 1D array
        if(len(x)>3):   # accept x of min length of 4 elements (M=2)
            # check length of signal, adjust to 2**M
            n=len(x)
            M=trunc(log(n,2))
            x=x[0:2**M]   # a trimmed signal
            a=(2**(M/2))  # a number of adjacent sequences/blocks
            b=2**M/a      # a number of elements in each sequence
            y=np.reshape(x,(a,b))  # (a x b) array of split sequences
            return (y,x,a,b,M)
        else:
            print("xsequences(x): Array too short!")
            raise SystemExit
    else:
        print("xsequences(x): 1D array expected!")
        raise SystemExit


def tstat(x,a,b,M):
    # specify the probability of occurrence of the digit "1"
    p=0.5
    print("Computation of WHTs...")
    for j in xrange(a):
        hwt, _, _ = WHT(x[j])
        if(j==0):
            y=hwt
        else:
            y=np.vstack((y,hwt))
    print("  ...completed")
    print("Computation of t-statistics..."),
    t=[];
    for j in xrange(a):     # over sequences/blocks (rows)
        for i in xrange(b): # over sequence's elements (columns)
            if(i==0):
                if(p==0.5):
                    m0j=0
                else:
                    m0j=(2.**M/2.)*(1.-2.*p)
                sig0j=sqrt((2**M/2)*p*(1.-p))
                w0j=y[j][i]
                t0j=(w0j-m0j)/sig0j
                t.append(t0j)
            else:
                sigij=sqrt((2.**((M+2.)/2.))*p*(1.-p))
                wij=y[j][i]
                tij=wij/sigij
                t.append(tij)
    t=np.array(t)
    print("completed")
    print("Computation of p-values..."),
    # standardised t-statistics; t_{i,j} ~ N(0,1)
    t=(t-np.mean(t))/(np.std(t))
    # p-values = 1-[1/sqrt(2*pi)*integral[exp(-x**2/2),x=-inf..t]]
    P=1-ndtr(t)
    print("completed\n")
    return(t,P,y)


def info(X,xt,a,b,M):
    line()
    print("Signal\t\tX(t)")
    print("  of length\tn = %d digits" % len(X))
    print("trimmed to\tx(t)")
    print("  of length\tn = %d digits (n=2^%d)" % (a*b,M))
    print("  split into\ta = %d sub-sequences " % a)
    print("\t\tb = %d-digit long" % b)
    print


def test1(cl,t,a,b,otest):
    alpha=1.-cl/100.
    u1=norm.ppf(alpha/2.)
    u2=norm.ppf(1-alpha/2.)
    Results1=[]
    for l in t:
        if(l<u1 or l>u2):
            Results1.append(0)
        else:
            Results1.append(1)
    nfail=a*b-np.sum(Results1)
    print("Test 1 (Crude Decision)")
    print("  RESULT: %d out of %d test variables stand for " \
"randomness" % (a*b-nfail,a*b))
    if((a*b-nfail)/float(a*b)>.99):
        print("\t  Signal x(t) appears to be random")
    else:
        print("\t  Signal x(t) appears to be non-random")
    otest.append(100.*(a*b-nfail)/float(a*b))
    print("\t  at %.5f%% confidence level" % (100.*(1.-alpha)))
    print
    return(otest)


def test2(cl,P,a,b,otest):
    alpha=1.-cl/100.
    u1=norm.ppf(alpha/2.)
    u2=norm.ppf(1-alpha/2.)
    Results2=[]
    rP=np.reshape(P,(a,b))
    for j in xrange(a):
        tmp=rP[j][(rP[j]<alpha)]
        #print(tmp)
        if(len(tmp)>0):
            Results2.append(0)   # fail for sub-sequence
        else:
            Results2.append(1)   # pass

    nfail2=a-np.sum(Results2)  # total number of sub-sequences which failed
    t2=nfail2/float(a)
    print("Test 2 (Proportion of Sequences Passing a Test)")
    b1=alpha*a+sqrt(a*alpha*(1-alpha))*u1
    b2=alpha*a+sqrt(a*alpha*(1-alpha))*u2
    if(t2<b1 or t2>b2):
        print("  RESULT: Signal x(t) appears to be non-random")
        otest.append(0.)
    else:
        print("  RESULT: Signal x(t) appears to be random")
        otest.append(100.)
    print("\t  at %.5f%% confidence level" % (100.*(1.-alpha)))
    print
    return(otest)

def test3(cl,P,a,b,otest):
    alpha=1.-cl/100.
    rP=np.reshape(P,(a,b))
    rPT=rP.T
    Results3=0
    for i in xrange(b):
        (hist,bin_edges,_)=plt.hist(rPT[i], bins=list(np.arange(0.0,1.1,0.1)))
        F=hist
        K=len(hist)  # K=10 for bins as defined above
        S=a
        chi2=0
        for j in xrange(K):
            chi2+=((F[j]-S/K)**2.)/(S/K)
        pvalue=1-gammainc(9/2.,chi2/2.)
        if(pvalue>=alpha and chi2<=Chi2(alpha,K-1)):
            Results3+=1
    print("Test 3 (Uniformity of p-values)")
    print("  RESULT: %d out of %d test variables stand for randomness"\
% (Results3,b))
    if((Results3/float(b))>.99):
        print("\t  Signal x(t) appears to be random")
    else:
        print("\t  Signal x(t) appears to be non-random")
    otest.append(100.*(Results3/float(b)))
    print("\t  at %.5f%% confidence level" % (100.*(1.-alpha)))
    print
    return(otest)


def test4(cl,t,a,b,otest):
    alpha=1.-cl/100.
    rt=np.reshape(t,(a,b))
    rtT=rt.T
    Results4=0
    for i in xrange(b):
        tmp=np.max(rtT[i])
        u1=norm.ppf((alpha/2.)**(1./a))
        u2=norm.ppf((1.-alpha/2.)**(1./a))
        if not(tmp<u1 or tmp>u2):
            Results4+=1
    print("Test 4 (Maximum Value Decision)")
    print("  RESULT: %d out of %d test variables stand for randomness" % (Results4,b))
    if((Results4/float(b))>.99):
        print("\t  Signal x(t) appears to be random")
    else:
        print("\t  Signal x(t) appears to be non-random")
    otest.append(100.*(Results4/float(b)))
    print("\t  at %.5f%% confidence level" % (100.*(1.-alpha)))
    print
    return(otest)


def test5(cl,t,a,b,otest):
    alpha=1.-cl/100.
    rt=np.reshape(t,(a,b))
    rtT=rt.T
    Results5=0
    for i in xrange(b):
        Ci=0
        for j in xrange(a):
           Ci+=(rtT[i][j])**2.
        if(Ci<=Chi2(alpha,a)):
            Results5+=1
    print("Test 5 (Sum of Square Decision)")
    print("  RESULT: %d out of %d test variables stand for randomness" % (Results5,b))
    if((Results5/float(b))>.99):
        print("\t  Signal x(t) appears to be random")
    else:
        print("\t  Signal x(t) appears to be non-random")
    otest.append(100.*(Results5/float(b)))
    print("\t  at %.5f%% confidence level" % (100.*(1.-alpha)))
    print
    return(otest)


def overalltest(cl,otest):
    alpha=1.-cl/100.
    line()
    print("THE OVERALL RESULT:")
    if(np.mean(otest)>=99.0):
        print("   Signal x(t) displays an evidence for RANDOMNESS"),
        T=1
    else:
        print("   Signal x(t) displays an evidence for NON-RANDOMNESS"),
        T=0
    print("at %.5f%% c.l." % (100.*(1.-alpha)))
    print("   based on Walsh-Hadamard Transform Statistical Test\n")
    return(T)


def WHTStatTest(cl,X):
    (xseq,xt,a,b,M) = xsequences(X)
    info(X,xt,a,b,M)
    if(M<7):
        line()
        print("Error:  Signal x(t) too short for WHT Statistical Test")
        print("        Acceptable minimal signal length: n=2^7=128\n")
    else:
        if(M>=7 and M<19):
            line()
            print("Warning: Statistically advisable signal length: n=2^19=524288\n")
        line()
        print("Test Name: Walsh-Hadamard Transform Statistical Test\n")
        (t, P, xseqWHT) = tstat(xseq,a,b,M)
        otest=test1(cl,t,a,b,[])
        otest=test2(cl,P,a,b,otest)
        otest=test3(cl,P,a,b,otest)
        otest=test4(cl,t,a,b,otest)
        otest=test5(cl,t,a,b,otest)
        T=overalltest(cl,otest)
        return(T)  # 1 if x(t) is random, else 0

def bit_reverse_traverse(a):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = a.shape[0]
    assert(not n&(n-1) ) # assert that n is a power of 2
    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(n/2)*2
        odd_index = np.arange(n/2)*2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd

def get_bit_reversed_list(l):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])
    return b

def FWHT(X):
    # Fast Walsh-Hadamard Transform for 1D signals
    # of length n=2^M only (non error-proof for now)
    x=get_bit_reversed_list(X)
    x=np.array(x)
    N=len(X)

    for i in range(0,N,2):
        x[i]=x[i]+x[i+1]
        x[i+1]=x[i]-2*x[i+1]

    L=1
    y=np.zeros_like(x)
    for n in range(2,int(log(N,2))+1):
        M=2**L
        J=0; K=0
        while(K<N):
            for j in range(J,J+M,2):
                y[K]   = x[j]   + x[j+M]
                y[K+1] = x[j]   - x[j+M]
                y[K+2] = x[j+1] + x[j+1+M]
                y[K+3] = x[j+1] - x[j+1+M]
                K=K+4
            J=J+2*M
        x=y.copy()
        L=L+1

    return x/float(N)
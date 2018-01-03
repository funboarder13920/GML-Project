import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

def EXP3(G, U, eta, gamma, T):
    # nodes of G must be ordered and separated by 1 [0 1 2 ...], [0 2 3] is forbidden
    V = list(G.nodes())
    t = 0
    u = np.array([0 if not n in U else 1/len(U) for n in V])
    q = (1/len(V))*np.ones((T+1, len(V)))
    p = np.zeros((T, len(V)))
    losses = np.zeros((T,))
    for t in tqdm(range(T), desc="Simulating EXP3"):
        p[t] = (1-gamma)*q[t]+gamma*u
        draw = np.random.multinomial(1, p[t])
        It = V[np.argmax(draw)]
        
        # observe
        loss = {action: G.node[action]['arm'].sample()/sum([p[t][pred] for pred in G.predecessors(action)]) for action in G.successors(It)}
        losses[t] = G.node[It]['arm'].sample()
        q[t+1] = np.array([q[t][i]*np.exp(-eta*loss[i]) if i in loss else q[t][i] for i in V])
        q[t+1] = 1/(sum(q[t+1]))*q[t+1]
    return q[-1], losses

def compute_regret(losses, G):
    n_itr = losses.shape[0]
    best_arm_mean = np.min([-G.node[node]['arm'].mean for node in G.nodes()]) 
    return np.cumsum(losses)-best_arm_mean*np.arange(1, n_itr + 1)

def plot_regret(values, labels, asympt=True, linReg=False, savefig=None):
    plt.figure()
    n_itr = values[0].shape[0]
    x = np.arange(1, n_itr+1)
    maxVal = 0
    for i in range(len(values)):
        plt.plot(x, values[i], label=labels[i])
        maxVal = max(maxVal, max(values[i]))
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    if asympt:
        plt.plot(x, x, label="No learning")
    if linReg:
        [der2,linAreas] = find_linear_areas(values[0], 0.01)
        ct = 1
        for linArea in linAreas:
            [slope, intercept] = lin_regression(values[0], linArea)
            print("Found linear domain {0:2d} at [{1:4d}, {2:4d}]: Slope {3:.2f}, intercept {4:.2f}".format(ct, linArea[0], linArea[-1], slope, intercept))
            plt.plot(x, intercept + slope*x, label="Linear domain {0:2d}".format(ct))
            ct = ct+1
    plt.legend()
    plt.ylim([0,maxVal])
    if savefig:
        plt.savefig(savefig)
    plt.show()
    return der2,linAreas

def find_linear_areas(values, thr, filt=True):
    der2 = np.ones(len(values))
    prevValues = np.roll(values,1)
    nextValues = np.roll(values,-1)
    der2 = prevValues + nextValues - 2*values
    n = len(der2) - 1
    if filt:
        der2 = gaussian_filter(der2)
        n = len(der2) - 1
    linTrend = (abs(der2)<thr)
    parsedLinTrendsAreas = []
    seg = []
    for ind in range(n):
        if linTrend[ind]:
            seg.append(ind)
        elif seg != []:
            parsedLinTrendsAreas.append(seg)
            seg = []
    return der2, parsedLinTrendsAreas
    
def lin_regression(values, idxRange):
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(values[idxRange])),values[idxRange])
    return slope, intercept

def gaussian_filter(values,strippedXs=False,degree=15,baseWeight=2.0):  
    window=degree*2-1  
    weight=np.array([baseWeight]*window)  
    weightGauss=[]  
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
        weight=np.array(weightGauss)*weight  
        smoothed=[0.0]*(len(values)-window)  
        for i in range(len(smoothed)):  
            smoothed[i]=sum(np.array(values[i:i+window])*weight)/sum(weight)  
        return np.array(smoothed)

def lower_bound():
    pass

def upper_bound():
    pass
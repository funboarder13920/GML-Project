import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from obsGraph import observability_type

def EXP3(G, U, eta, gamma, T=5000, n_sim=50, perturbations=None):
    # nodes of G must be ordered and separated by 1 [0 1 2 ...], [0 2 3] is forbidden
    # perturbations is a dictionnary, mapping perturbation times to lists of edges to cut
    if perturbations is None:
        perturbations = {}
    V = list(G.nodes())
    avg_losses = np.zeros((T,))
    avg_q = np.zeros((T+1, len(V)))
    
    for sim in range(n_sim):
        t = 0
        u = np.array([0 if not n in U else 1/len(U) for n in V])
        q = (1/len(V))*np.ones((T+1, len(V)))
        p = np.zeros((T, len(V)))
        losses = np.zeros((T,))
        for t in tqdm(range(T), desc="Simulating EXP3"):
        
            edges = perturbations.get(t,[])
            for edge in edges:
                G.remove_edge(edge[0], edge[1])
                print("Edge {0} removed at iteration {1}".format(edge, t))
                obs_dict = {0:"unobservable", 1:"weakly observable", 2:"strongly observable"}
                obs_type = observability_type(G)
                print("G is {}".format(obs_dict[obs_type]))
            p[t] = (1-gamma)*q[t]+gamma*u
            draw = np.random.multinomial(1, p[t])
            It = V[np.argmax(draw)]
        
        # observe
            loss = {
                action: G.node[action]['arm'].sample()/sum(
                    [p[t][pred] for pred in G.predecessors(action)]
                ) for action in G.successors(It)
            }
            losses[t] = G.node[It]['arm'].sample()
            q[t+1] = np.array([q[t][i]*np.exp(-eta*loss[i]) if i in loss else q[t][i] for i in V])
            q[t+1] = 1/(sum(q[t+1]))*q[t+1]
        avg_losses = avg_losses + (1.0/n_sim)*losses
        avg_q = avg_q + (1.0/n_sim)*q
    return avg_q[-1], avg_losses

def compute_regret(losses, G):
    n_itr = losses.shape[0]
    best_arm_mean = np.min([-G.node[node]['arm'].mean for node in G.nodes()]) 
    return np.cumsum(losses)-best_arm_mean*np.arange(1, n_itr + 1)

def plot_regret(values, labels, asympt=True, reg="", savefig=None, stdev=15):
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
    
    der2 = []
    linAreas = []
    
    if reg != "":
        regValues = values
        threshold = 0.06
        if reg == "Pwr1/2":
            regValues = np.power(values,2)
            threshold = np.sqrt(threshold)
        elif reg == "Pwr2/3":
            regValues = np.power(values,3.0/2)
            threshold = np.power(threshold,2.0/3)
            
        [der2,linAreas] = find_linear_areas(regValues[0], threshold, stdev=stdev)
        ct = 1
        for linArea in linAreas:
            [slope, intercept] = lin_regression(regValues[0], linArea)
            print(
                "Found {0:s} domain {1:2d} at [{2:4d}, {3:4d}]: Slope {4:.2f}, intercept {5:.2f}".format(
                    reg,
                    ct,
                    linArea[0],
                    linArea[-1],
                    slope,
                    intercept-slope*linArea[0]
                )
            )
            y = intercept + slope*(x-linArea[0])
            if reg == "Pwr1/2":
                y = np.sqrt(np.clip(y,0,1e+30))
            elif reg == "Pwr2/3":
                y = np.power(np.clip(y,0,1e+30),2.0/3)
            plt.plot(x, y, label="{0:s} domain {1:2d}".format(reg, ct))
            ct = ct+1     
    
    plt.legend()
    plt.ylim([0,maxVal])
    if savefig:
        plt.savefig(savefig)
    plt.show()
    return der2,linAreas

def find_linear_areas(values, thr, filt=True, stdev=15):
    der2 = np.ones(len(values))
    prevValues = np.roll(values,1)
    nextValues = np.roll(values,-1)
    der2 = prevValues + nextValues - 2*values
    n = len(der2) - 1
    if filt:
        der2 = gaussian_filter(der2, degree=stdev)
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

def gaussian_filter(values,strippedXs=False,degree=15):  
    window=degree*2-1  
    weight=np.array([1.0]*window)  
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
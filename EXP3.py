import numpy as np
import pdb
from scipy import stats
from tqdm import tqdm
from networkx.algorithms.approximation import *
import matplotlib.pyplot as plt
from obsGraph import observability_type, weak_dom_number

def EXP3(G, U, eta, gamma, T=5000, n_sim=50, perturbations=None):
    # nodes of G must be ordered and separated by 1 [0 1 2 ...], [0 2 3] is forbidden
    # perturbations is a dictionnary, mapping perturbation times to lists of edges to cut  
    obs_dict = {0:"unobservable", 1:"weakly observable", 2:"strongly observable"}
    obs_type = observability_type(G)
    # print("G is {}".format(obs_dict[obs_type]))
    
    if perturbations is None:
        perturbations = {}
    V = list(G.nodes())
    avg_losses = np.zeros((T,))
    avg_q = np.zeros((T+1, len(V)))
    
    for sim in tqdm(range(n_sim), desc="Simulating EXP3 on {} runs".format(n_sim)):
        H = G.copy()
        t = 0
        u = np.array([0 if not n in U else 1/len(U) for n in V])
        q = (1/len(V))*np.ones((T+1, len(V)))
        p = np.zeros((T, len(V)))
        losses = np.zeros((T,))
        for t in range(T):
        
            edges = perturbations.get(t,[])
            for edge in edges:
                if edge in list(H.edges()):
                    H.remove_edge(edge[0], edge[1])
                    print("Edge {0} removed at iteration {1}".format(edge, t))
                    obs_type = observability_type(H)
                    print("G is {}".format(obs_dict[obs_type]))
                else:
                    print("Edge is already missing from graph")
            p[t] = (1-gamma)*q[t]+gamma*u
            draw = np.random.multinomial(1, p[t])
            It = V[np.argmax(draw)]
        
        # observe
            loss = {
                action: H.node[action]['arm'].sample()/sum(
                    [p[t][pred] for pred in H.predecessors(action)]
                ) for action in H.successors(It)
            }
            losses[t] = H.node[It]['arm'].sample()
            q[t+1] = np.array(
                [q[t][i]*np.exp(-eta*loss[i]) if i in loss else q[t][i] for i in V]
            )
            q[t+1] = 1/(sum(q[t+1]))*q[t+1]
        avg_losses = avg_losses + (1.0/n_sim)*losses
        avg_q = avg_q + (1.0/n_sim)*q
    return avg_q[-1], avg_losses


def EXP3Opt(G, U, T=5000, n_sim=50, perturbations=None, alpha=None, delta=None):
    # nodes of G must be ordered and separated by 1 [0 1 2 ...], [0 2 3] is forbidden
    obs_dict = {0:"unobservable", 1:"weakly observable", 2:"strongly observable"}
    obs_type = observability_type(G)
    
    #1) If no independence number is provided, find an approximation of it, only for small graphs
    K = G.number_of_nodes()
    if obs_type == 2 and alpha is None:
        max_ind_set = independent_set.maximum_independent_set(G)
        alpha = len(max_ind_set)
    
    if obs_type == 1 and delta is None:
        delta = weak_dom_number(G)

    #2) Determining algorithm constants
    if obs_type == 2:
        gamma = min(np.sqrt(1/(alpha*T)), 0.5)
        eta = 2*gamma
    else:
        gamma = min(np.power(delta*np.log(K)/T, 1.0/3),0.5)
        eta = np.power(gamma,2)/delta
    
    #NB) We do not handle changes in parameters in case the graph observability graph changes
    
    #3) Run simulations
    return EXP3(G, U, eta, gamma, T, n_sim, perturbations)


def compute_regret(losses, G):
    n_itr = losses.shape[0]
    best_arm_mean = np.min([-G.node[node]['arm'].mean for node in G.nodes()]) 
    return np.cumsum(losses)-best_arm_mean*np.arange(1, n_itr + 1)

def plot_regret(G, values, labels, asympt=True, reg="", savefig=None, stdev=15):
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
        best_arm_mean = np.min([-G.node[node]['arm'].mean for node in G.nodes()])
        avg_arm_mean = np.mean([-G.node[node]['arm'].mean for node in G.nodes()])
        plt.plot(x, x*(avg_arm_mean - best_arm_mean), label="No learning")
    
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

def upper_bound(G, T, alpha=None, delta=None):
    obs_type = observability_type(G)
    if obs_type == 2:
        if alpha is None:
            max_ind_set = independent_set.maximum_independent_set(G)
            alpha = len(max_ind_set)
        return alpha**(1/2)*np.array([t**(1/2)*np.log((1+t)*G.number_of_nodes()) for t in range(T)])
    if obs_type == 1:
        if delta is None:
            delta = weak_dom_number(G)
        return (delta*np.log(G.number_of_nodes()))**(1/3)*np.array([t**(2/3) for t in range(T)])
    if obs_type == 0:
        return np.array([t for t in range(T)])

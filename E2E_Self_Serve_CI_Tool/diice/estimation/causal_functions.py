import pandas as pd
import numpy as np

def ω(g, t, clip_bounds=(1e-3, 1 - 1e-3)):
    g = g.clip(*clip_bounds)
    return (t/g) + ((1-t)/(1-g))


def τ_se(model, y, x_where_t_1, x_where_t_0, cate, t):
    p = t.mean()
    var_tau = np.var(cate)
    try:
        yhat_given_1 = model.predict_proba(x_where_t_1)[:,1]
        yhat_given_0 = model.predict_proba(x_where_t_0)[:,1]
    except:
        yhat_given_1 = model.predict(x_where_t_1)
        yhat_given_0 = model.predict(x_where_t_0)   
    var_treatment = np.var(y[t==1] - yhat_given_1)
    print('var_treatment:', var_treatment)
    var_control = np.var(y[t==0] - yhat_given_0)
    print('var_control:', var_control)
    return np.sqrt(
        (var_tau + (var_treatment / p) + (var_control / (1-p))) / len(y)
    )


def plot_heterogenity(cate, yhat=None, treatment=None, bins=100, title=None):
    cate.sort()
    window_size = len(cate) // bins
    
    mean_cate, se_cate, percentile = [], [], []
    for window in range(bins):
        window += 1
        cum_cate = cate[-window * window_size:]
        mean_cate.append(cum_cate.mean())
        se_cate.append(cum_cate.std() / np.sqrt(len(cum_cate)))
        percentile.append(window / bins)
    mean_cate, se_cate, percentile = [np.array(a) for a in (mean_cate, se_cate, percentile)]
    
    # Create the fancy plot
    graph.figure(figsize=(5, 4))
    graph.title('(Targeting Operating Characteristic)' if title is None else f'{title} (TOC)')
    graph.plot(percentile, mean_cate, linewidth=1, label='')
    graph.fill_between(
        percentile, 
        mean_cate - (2.56*se_cate), 
        mean_cate + (2.56*se_cate),
        alpha=0.33
    )
    graph.axhline(cate.mean(), linewidth=1, color='black', label='Average $\psi$')
    graph.ylabel('$\psi$')
    graph.xlabel('Cumulative Percent')
    graph.legend()
    graph.xlim([0, 1])
    graph.show()

    
def get_potential_outcome_df(data, treatment, features):
    x0 = data[features + [treatment]].copy()
    x0[treatment] = 0
    
    x1 = x0.copy()
    x1[treatment] = 1
    
    return x0, x1
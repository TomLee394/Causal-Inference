def ω(g, t, clip_bounds=(1e-3, 1 - 1e-3)):
    g = g.clip(*clip_bounds)
    return (t/g) + ((1-t)/(1-g))


def τ_se(y, yhat_given_1, yhat_given_0, cate, t):
    p = t.mean()
    var_tau = np.var(cate)
    var_treatment = np.var(y[t==1] - yhat_given_1[t==1])
    var_control = np.var(y[t==0] - yhat_given_0[t==0])
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

    
def get_potential_outcome_df(d: pd.DataFrame):
    x0 = d[features_list_decollinear + ['treatment_status']].copy()
    x0['treatment_status'] = 0
    
    x1 = x0.copy()
    x1['treatment_status'] = 1
    
    return x0, x1
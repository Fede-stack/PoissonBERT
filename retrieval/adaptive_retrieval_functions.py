def return_kstar(data, embeddings, initial_id=None, Dthr=12, r='opt', n_iter = 10):
  #return id estimate together with kstar neighbors for each observation
    if initial_id is None:
        data.compute_id_2NN(algorithm='base')
    else:
        data.compute_distances()
        data.set_id(initial_id)

    ids = np.zeros(n_iter)
    ids_err = np.zeros(n_iter)
    kstars = np.zeros((n_iter, data.N), dtype=int)
    log_likelihoods = np.zeros(n_iter)
    ks_stats = np.zeros(n_iter)
    p_values = np.zeros(n_iter)

    for i in range(n_iter):
      #compute kstar
      data.compute_kstar(Dthr)
      

      #set new ratio
      r_eff = min(0.95,0.2032**(1./data.intrinsic_dim)) if r == 'opt' else r
      #compute neighbourhoods shells from k_star
      rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
      rn = rk * r_eff
      n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
      #compute id
      id = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
      #compute id error
      id_err = ut._compute_binomial_cramerrao(id, data.kstar-1, r_eff, data.N)
      #compute likelihood
      log_lik = ut.binomial_loglik(id, data.kstar - 1, n - 1, r_eff)
      #model validation through KS test
      n_model = rng.binomial(data.kstar-1, r_eff**id, size=len(n))
      ks, pv = ks_2samp(n-1, n_model)
      #set new id
      data.set_id(id)

      ids[i] = id
      ids_err[i] = id_err
      kstars[i] = data.kstar
      log_likelihoods[i] = log_lik
      ks_stats[i] = ks
      p_values[i] = pv

    data.intrinsic_dim = id
    data.intrinsic_dim_err = id_err
    data.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())

    return ids, kstars[(n_iter - 1), :]#, ids_err, log_likelihoods, ks_stats, p_values

def find_Kstar_neighs(kstars, embeddings):
  #return the nearest neighbors for each observation in the dataset
    nn = NearestNeighbors(metric = 'cosine', n_jobs=-1)
    nn.fit(embeddings)

    neighs_ind = []
    for i, obs in enumerate(embeddings):
        distance, ind = nn.kneighbors([obs], n_neighbors=kstars[i] + 1)

        k_neighs = ind[0][1:]
        neighs_ind.append(k_neighs.tolist())
    return neighs_ind


def find_single_k_neighs(embeddings, index, k):
  #focus only on query kstars neighbors. Given an array of embeddings, it requires the index of the query and the list of kstar neighbors return by return_kstar function
    target_embedding = embeddings[index]
    all_distances = np.array([distance.cosine(target_embedding, emb) for emb in embeddings])

    nearest_indices = np.argsort(all_distances)[1:k+1]  # +1 per saltare l'osservazione stessa

    return nearest_indices.tolist()

def extract_docs(results):

    unique_set = set()
    for sublist in results:
        unique_set.update(sublist)
    unique_docs = list(unique_set)

    return unique_docs

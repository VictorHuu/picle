import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from data.formatting import format_icl_queries, statement_from_question
from models.scoring import embed_statements


def return_indices(num_clusters, km, train_embs):
    indices = []
    for cluster in range(num_clusters):
        members = np.flatnonzero(km.labels_ == cluster)
        if not len(members):
            raise ValueError("Diversity selection produced an empty cluster; reduce K.")
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_[cluster:cluster + 1], train_embs[members])
        indices.append(int(members[closest[0]]))
    return indices


def get_embeddings(args, dataset, ref_model, statements=None):
    statements = statements if statements is not None else [statement_from_question(x) for x in dataset]
    return embed_statements(ref_model, statements, layer=-1)


def select_diverse(args, test, test_labels, train, train_labels, K=3, ref_model=None, train_statements=None):
    embeddings = get_embeddings(args, train, ref_model, train_statements).numpy()
    km = KMeans(n_clusters=K, n_init="auto", random_state=args.seed).fit(embeddings)
    selected = return_indices(K, km, embeddings)
    return format_icl_queries(args, test, test_labels, train, train_labels, [selected] * len(test), ref_model)

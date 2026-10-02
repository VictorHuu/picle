from data.formatting import format_icl_queries
from icl_strategies.select_similar import get_embeddings
from icl_strategies.select_uncertain import get_uncertainties


def select_certain_and_similar(args, test, test_labels, train, train_labels, K=3, N=10,
                               choose_uncertain=False, func="bin_entropy", ref_model=None):
    scores = get_uncertainties(args, train, train_labels, ref_model, func)
    if not choose_uncertain:
        scores = -scores
    if not K <= N <= len(train):
        raise ValueError("Expected K <= N <= candidate count.")
    candidates = scores.topk(N).indices.tolist()
    pool = [train[i] for i in candidates]
    pool_labels = [train_labels[i] for i in candidates]
    train_embs = get_embeddings(args, pool, ref_model)
    test_embs = get_embeddings(args, test, ref_model)
    selections = (test_embs @ train_embs.T).topk(K).indices.flip(1).tolist()
    return format_icl_queries(args, test, test_labels, pool, pool_labels, selections, ref_model)

from data.formatting import format_icl_queries, statement_from_question
from models.scoring import score_features, statement_feature


def get_likelihood(args, dataset, ref_model, train_statements=None):
    statements = train_statements if train_statements is not None else [statement_from_question(x) for x in dataset]
    return score_features(ref_model, [statement_feature(ref_model, x + ".\n") for x in statements])


def select_likely(args, test, test_labels, train, train_labels, K, ref_model, train_statements=None):
    likelihood = get_likelihood(args, train, ref_model, train_statements)
    selected = likelihood.topk(K).indices.flip(0).tolist()
    return format_icl_queries(args, test, test_labels, train, train_labels, [selected] * len(test), ref_model)

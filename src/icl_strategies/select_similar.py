from data.formatting import format_icl_queries, statement_from_question
from models.scoring import embed_statements


def get_embeddings(args, dataset, ref_model, statements=None):
    if statements is None:
        statements = [statement_from_question(x) for x in dataset]
    layer = -0.5 if args.midlayer_for_sim else (-2 if args.penultlayer_for_sim else -1)
    return embed_statements(ref_model, statements, layer)


def select_similar(args, test, test_labels, train, train_labels, K=3, ref_model=None,
                   train_statements=None, test_statements=None):
    train_embs = get_embeddings(args, train, ref_model, train_statements)
    test_embs = get_embeddings(args, test, ref_model, test_statements)
    # Original baseline: last-token hidden states and unnormalized dot product.
    indices = (test_embs @ train_embs.T).topk(K).indices.flip(1).tolist()
    return format_icl_queries(args, test, test_labels, train, train_labels, indices, ref_model)

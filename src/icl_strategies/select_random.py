import random

from data.formatting import format_icl_queries


def select_random(args, test, test_labels, train, train_labels, K=3, ref_model=None):
    rng = random.Random(args.seed)
    # Preserve sampling with replacement from the original implementation.
    selections = [rng.choices(range(len(train)), k=K) for _ in test]
    return format_icl_queries(args, test, test_labels, train, train_labels, selections, ref_model)

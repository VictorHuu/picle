from data.formatting import format_icl_queries, label_to_int


def select_high_conf_label(args, test, test_labels, train, train_labels, K=3, ref_model=None):
    selected = [i for i, label in enumerate(train_labels) if label_to_int(label)][:K]
    if len(selected) != K:
        raise ValueError("Not enough positive-label demonstrations.")
    return format_icl_queries(args, test, test_labels, train, train_labels, [selected] * len(test), ref_model)

import torch
from tqdm import tqdm

from data.formatting import format_icl_queries, format_query
from models.scoring import score_answers


def get_uncertainties(args, queries, labels, ref_model, func):
    scores = []
    for question in tqdm(queries, desc="Answer uncertainty"):
        prompt = format_query(args, question, ref_model.tokenizer)
        answer_logp, token_logp = score_answers(ref_model, prompt)
        if func == "bin_entropy":
            logp = answer_logp.log_softmax(-1)
            score = -(logp.exp() * logp).sum()
        elif func == "cat_entropy":
            score = -(token_logp.exp() * token_logp).sum()
        elif func == "confidence":
            score = 1 - answer_logp.exp().max()
        else:
            raise ValueError(f"Unknown uncertainty function: {func}")
        scores.append(score)
    return torch.stack(scores)


def select_uncertain(args, test, test_labels, train, train_labels, K=3,
                     choose_uncertain=True, func="bin_entropy", ref_model=None):
    scores = get_uncertainties(args, train, train_labels, ref_model, func)
    if not choose_uncertain:
        scores = -scores
    selected = scores.topk(K).indices.flip(0).tolist()
    return format_icl_queries(args, test, test_labels, train, train_labels, [selected] * len(test), ref_model)

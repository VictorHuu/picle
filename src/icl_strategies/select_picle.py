import csv
from pathlib import Path

from data.formatting import format_icl_queries, statement_from_question
from models.scoring import base_model_context, score_features, statement_feature


def get_likelihood(args, dataset, ref_model, sft_model=None, func="diff", train_statements=None):
    statements = train_statements if train_statements is not None else [statement_from_question(x) for x in dataset]
    features = [statement_feature(ref_model, statement + ".\n") for statement in statements]
    # Reuse exactly the same token IDs and masks, switching only the adapter.
    # Base scores stay on CPU; no second copy of the 7B model is loaded.
    with base_model_context(ref_model):
        base_scores = score_features(ref_model, features, "Base likelihood")
    persona_model = sft_model if sft_model is not None else ref_model
    sft_scores = score_features(persona_model, features, "Persona likelihood")
    scores = sft_scores - base_scores if func == "diff" else sft_scores
    return scores, base_scores, sft_scores


def select_picle(args, test, test_labels, train, train_labels, K, func, ref_model,
                 sft_model=None, train_statements=None, train_ids=None):
    scores, base_scores, sft_scores = get_likelihood(
        args, train, ref_model, sft_model, func, train_statements,
    )
    # Highest-scoring demonstration comes closest to the test query.
    selected = scores.topk(K).indices.flip(0).tolist()
    folder = Path(args.results_dir) / args.target_persona
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "picle_selection.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "statement", "answer", "base_logp", "sft_logp", "delta", "context_position"])
        for i, question in enumerate(train):
            statement = train_statements[i] if train_statements is not None else statement_from_question(question)
            writer.writerow([train_ids[i] if train_ids is not None else i, statement, train_labels[i].strip(),
                             base_scores[i].item(), sft_scores[i].item(), (sft_scores[i] - base_scores[i]).item(),
                             selected.index(i) + 1 if i in selected else ""])
    return format_icl_queries(args, test, test_labels, train, train_labels, [selected] * len(test), ref_model)

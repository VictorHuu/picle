"""Prompt formatting shared by the evaluation and selection methods."""

SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def label_to_int(label):
    value = label.strip().lower()
    if value not in ("yes", "no"):
        raise ValueError(f"Expected a Yes/No label, got {label!r}")
    return int(value == "yes")


def format_query(args, question, tokenizer=None, demonstrations=()):
    instruction = question + ". Answer with Yes or No only."
    if args.model == "qwen":
        if tokenizer is None:
            raise ValueError("Qwen prompts require the model tokenizer.")
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        for query, answer in demonstrations:
            messages.extend([
                {"role": "user", "content": query + ". Answer with Yes or No only."},
                {"role": "assistant", "content": "Yes" if label_to_int(answer) else "No"},
            ])
        messages.append({"role": "user", "content": instruction})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.model == "vicuna":
        prefix = "".join(f"USER: {q}\nASSISTANT: {a.strip()}. </s>\n" for q, a in demonstrations)
        return prefix + f"USER: {instruction}\nASSISTANT:"
    if args.model == "gptj":
        prefix = "".join(f"Question: {q}\nAnswer: {a.strip()}.\n\n" for q, a in demonstrations)
        return prefix + f"Question: {instruction}\nAnswer:"
    prefix = "".join(f"<s> [INST] {q} [/INST] {a.strip()}. </s> " for q, a in demonstrations)
    return prefix + ("<s> " if demonstrations else "") + f"[INST] {instruction} [/INST]"


def format_icl_queries(args, test, test_labels, train, train_labels, selections, ref_model):
    queries = []
    for question, indices in zip(test, selections):
        demos = [(train[int(i)], train_labels[int(i)]) for i in indices]
        queries.append(format_query(args, question, ref_model.tokenizer, demos))
    return queries, [label_to_int(label) for label in test_labels]


def statement_from_question(question):
    """Compatibility for callers without the dataset's explicit statement field."""
    _, separator, statement = question.partition("\n")
    if not separator:
        raise ValueError("Expected a quoted statement after the question's first newline.")
    statement = statement.strip()
    if len(statement) >= 2 and statement[0] == statement[-1] == '"':
        return statement[1:-1]
    raise ValueError("Cannot extract the statement; pass the dataset statement field instead.")

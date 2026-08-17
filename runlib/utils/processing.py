from nltk.tokenize import sent_tokenize


def mark_name(name, article_content):
    tokens = article_content.replace("u\xa0", " ").split(" ")  # original paper preprocessor
    for start, end in name["offsets"]:
        end = end - 1
        if end - start > 0 or len(tokens[start]) > 1:
            tokens[start] = "[START] " + tokens[start]
            tokens[end] = tokens[end] + " [END]"
    return " ".join(tokens)


def sentences_with_name(name, article_content, replace=True):
    marked = mark_name(name, article_content)
    sentences = sent_tokenize(marked)
    if replace:
        return [
            s.replace("[START]", "").replace("[END]", "").strip().replace("  ", " ")
            for s in sentences
            if "[START]" in s or "[END]" in s
        ]
    return sentences

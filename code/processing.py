import nltk
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json

def make_names_dict(data):
    return [{key: value for key, value in record.items() if key not in ('names', 'ids', 'offsets', 'id_lens')} | {
        'names': [{
            'name': name,
            'ids': id_list,
            'offsets': offset_list
        } for name, id_list, offset_list in zip(record['names'],
                                                record['ids'],
                                                [fix_offsets(offset_list) for offset_list in record['offsets']])]
    } for record in data]


def fix_offsets(offsets):
    offsets = offsets[1:-1]
    offsets = offsets.replace("'\n '", "' '")
    offsets = offsets.replace("'", '')
    offsets = offsets.replace(', ', ',')
    offsets = offsets.replace('],[', ']|[')
    offsets = offsets.split('|')
    for i in range(len(offsets)):
        offsets[i] = list(map(int, offsets[i][1:-1].split(',')))
    return offsets


def mark_name(name, article_content):
    tokens = article_content.replace('u\xa0', u' ').split(' ')

    for offset in name['offsets']:
        start = offset[0]
        end = offset[1] - 1

        if end - start > 0 or len(tokens[start]) > 1:

            tokens[start] = '[START] ' + tokens[start]
            tokens[end] = tokens[end] + ' [END]'

    return ' '.join(tokens)


def sentences_with_name(name, article_content, replace=True):
    marked_content = mark_name(name, article_content)
    sentences = sent_tokenize(marked_content)
    if replace:
        sentences = [sentence.replace('[START]', '').replace('[END]', '').strip().replace('  ', ' ') if replace else sentence
                     for sentence in sentences
                     if '[START]' in sentence
                     or '[END]' in sentence]
    return sentences
import utils.utils
import utils.dataset_utils
import os
from tqdm import tqdm
import random
import nltk
import argparse


def correct_file_name(target_filename, target_path):
    candidate_filenames = os.listdir(target_path)
    best_match = None
    min_distance = float('inf')
    for candidate_filename in candidate_filenames:
        distance = edit_distance(target_filename, candidate_filename)
        if distance < min_distance:
            min_distance = distance
            best_match = candidate_filename
    print('\nchange path:', os.path.join(target_path, target_filename),'  to  ', os.path.join(target_path, best_match))
    os.rename(os.path.join(target_path, best_match), os.path.join(target_path, target_filename))


def get_text(qad, domain):
    qad['Filename'] = qad['Filename'].replace(':', '_').replace('?','').replace('*', '_')
    local_file = os.path.join(args.web_dir, qad['Filename']) if domain == 'SearchResults' \
        else os.path.join(args.wikipedia_dir, qad['Filename'])
    if not os.path.exists(local_file):
        correct_file_name(qad['Filename'], args.web_dir if domain == 'SearchResults' else args.wikipedia_dir)
    return get_file_contents(local_file, encoding='utf-8')


def select_relevant_portion(text):
    paras = text.split('\n')
    selected = []
    done = False
    for para in paras:
        sents = sent_tokenize.tokenize(para)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                selected.append(word)
                if len(selected) >= args.max_num_tokens:
                    done = True
                    break
            if done:
                break
        if done:
            break
        selected.append('\n')
    st = ' '.join(selected).strip()
    return st


def add_triple_data(datum, page, domain):
    qad = {'Source': domain}
    for key in ['QuestionId', 'Question', 'Answer']:
        qad[key] = datum[key]
    for key in page:
        qad[key] = page[key]
    return qad


def get_qad_triples(data):
    qad_triples = []
    for datum in data['Data']:
        for key in ['EntityPages', 'SearchResults']:
            for page in datum.get(key, []):
                qad = add_triple_data(datum, page, key)
                qad_triples.append(qad)
    return qad_triples


def convert_to_squad_format(qa_json_file, squad_file):
    qa_json = utils.dataset_utils.read_triviaqa_data(qa_json_file)
    qad_triples = get_qad_triples(qa_json)

    random.seed(args.seed)
    random.shuffle(qad_triples)

    data = []
    for qad in tqdm(qad_triples):
        qid = qad['QuestionId']

        text = get_text(qad, qad['Source'])
        selected_text = select_relevant_portion(text)

        question = qad['Question']
        para = {'context': selected_text, 'qas': [{'question': question, 'answers': []}]}
        data.append({'paragraphs': [para]})
        qa = para['qas'][0]
        qa['id'] = utils.dataset_utils.get_question_doc_string(qid, qad['Filename'])
        qa['qid'] = qid

        ans_string, index = utils.dataset_utils.answer_index_in_document(qad['Answer'], selected_text)
        if index == -1:
            if qa_json['Split'] == 'train':
                continue
        else:
            qa['answers'].append({'text': ans_string, 'answer_start': index})

        if qa_json['Split'] == 'train' and len(data) >= args.sample_size and qa_json['Domain'] == 'Web':
            break

    squad = {'data': data, 'version': qa_json['Version']}
    utils.utils.write_json_to_file(squad, squad_file)
    print ('Added', len(data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triviaqa_file', help='Triviaqa file')
    parser.add_argument('--squad_file', help='Squad file')
    parser.add_argument('--wikipedia_dir', help='Wikipedia doc dir')
    parser.add_argument('--web_dir', help='Web doc dir')

    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--max_num_tokens', default=800, type=int, help='Maximum number of tokens from a document')
    parser.add_argument('--sample_size', default=80000, type=int, help='Random seed')
    parser.add_argument('--tokenizer', default='tokenizers/punkt/english.pickle', help='Sentence tokenizer')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    sent_tokenize = nltk.data.load(args.tokenizer)
    convert_to_squad_format(args.triviaqa_file, args.squad_file)

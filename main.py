import json
from transformers import AutoTokenizer
import spacy
import random
import os
from evaluate_model import run_model, stats_precisions_recalls
from plot_model import eval_model

is_chance = lambda chance: random.random() <= chance


filters = [
    {
        "name": "MaskAllProperNouns",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": None,
    },
    # { # probabilistic filter
    #     "name": "MaskAllProperNounsProbabilistic",
    #     "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent if is_chance(0.5)],
    #     "condition": lambda ent, doc: True,
    #     "mask_with": None,
    #     "num_runs": 6,
    # },
    {
        "name": "MaskPersonAndGPEProperNouns",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent if ent.label_ == "PERSON" or ent.label_ == "GPE"],
        "mask_with": None,
        "num_runs": None,
        # .text
    },
    {
        "name": "MaskAllNouns",
        "get_tokens": lambda doc: doc,
        "filtered_tokens": lambda doc: [token.i for token in doc if token.pos_ == "NOUN"],
        "mask_with": None,
        "num_runs": None,
    },
]

TEST_FILENAME = "./data/test.json"
TEST_FILENAME_TEMPLATE = "./data/test{}.json"
TEST_BACKUP_FILENAME = "./data/test.bak.json"
MODEL_NAME = 'roberta-base'
PATH_TO_PYTHON_PROJECT = os.getcwd()
PATH_TO_VENV = os.path.join(PATH_TO_PYTHON_PROJECT, '.env', 'Scripts', 'python')
STATS_FILENAME = 'stats.json'

# Using spaCy for NER
nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def mask_entities(context):
    doc = nlp(context)
    for ent in doc.ents:
        # if ent.label_ == "PERSON" or ent.label_ == "GPE":  # These are the entity types for proper nouns and geopolitical entities respectively
        context = context.replace(ent.text, tokenizer.mask_token)
    return context


def read_test_file(filename=TEST_FILENAME):
    # Load json file
    with open(filename, 'r') as file:
        data = json.load(file)
        return data


def write_test_file(data, filename=TEST_FILENAME):
    with open(filename, 'w') as file:
        json.dump(data, file)


def backup_test_file(data, backup_filename=TEST_BACKUP_FILENAME):
    with open(backup_filename, 'w') as file:
        json.dump(data, file)


def unmask_documents(flt):
    os.replace(TEST_FILENAME, TEST_FILENAME_TEMPLATE.format(f'_{flt["name"]}.masked'))
    os.replace(TEST_BACKUP_FILENAME, TEST_FILENAME)



def construct_answer_helper(document):
    # answer_positions_helper = [
    #     [
    #         {
    #             "text": "SUPPLY CONTRACT",
    #             "answer_start": 14
    #         }
    #     ],
    #     [
    #         {
    #             "answer_start": 143,
    #             "answer_end": 200
    #             "answer_len": 57
    #         },
    #         {
    #             "answer_start": 49,
    #             "answer_end": 100
    #             "answer_len": 51
    #         }
    #     ],
    #     ...
    # ]
    questions_and_answers = document['qas']
    answer_positions_helper = []

    for qas in questions_and_answers:
        answers = []
        for answer in qas['answers']:
            answer_pos = {}
            answer_pos['answer_start'] = answer['answer_start']
            answer_pos['answer_end'] = answer['answer_start'] + len(answer['text'])
            answer_pos['answer_len'] = len(answer['text'])
            answers.append(answer_pos)
        answer_positions_helper.append(answers)

    return answer_positions_helper


def mask_token_in_text(text, position_start_in_token, position_end_in_token, replace_token):
    text_after_masking = text[:position_start_in_token] + replace_token + text[position_end_in_token:]
    return text_after_masking


def mask_answers(document, answer_positions_helper, token_start, token_end, replace_token):
    questions_and_answers = document['qas']

    def is_token_in_answer(answer_start, answer_end):
        return (answer_start <= token_start <= answer_end) and (answer_start <= token_end <= answer_end)

    for helper, qas in zip(answer_positions_helper, questions_and_answers):
        answers = qas['answers']
        for answer_helper, answer in zip(helper, answers):
            answer_start = answer_helper['answer_start']
            answer_end = answer_helper['answer_end']

            if is_token_in_answer(answer_start, answer_end):
                # answer is in the token
                # replace token in the answer with a mask
                position_start_in_token = token_start - answer_start
                position_end_in_token = token_end - answer_start
                answer['text'] = mask_token_in_text(answer['text'], position_start_in_token, position_end_in_token, replace_token)

                # answer_helper['answer_len'] = len(answer['text'])
                # answer_helper['answer_end'] = answer_helper['answer_start'] + answer_helper['answer_len']


def update_answer_positions(document):
    context = document['context']
    questions_and_answers = document['qas']

    for qas in questions_and_answers:
        answers = qas['answers']
        for answer in answers:
            new_start = context.find(answer['text'])
            if new_start == -1:
                print("ERROR: answer not found in context")
                raise Exception("Answer not found in context")
            answer['answer_start'] = new_start


def mask_document(document, flt):
    # mutates doc
    context = document['context']
    doc = nlp(context)

    tokens = [token for token in doc]
    filtered_token_indicies = flt['filtered_tokens'](doc)
    answer_positions_helper = construct_answer_helper(document)
    replace_token = flt['mask_with'] or tokenizer.mask_token

    for token in reversed(tokens):
        i = token.i
        if i not in filtered_token_indicies:
            continue
        print(token.text)
        token_start = token.idx
        token_end = token.idx + len(token.text)

        # mutates doc
        mask_answers(document, answer_positions_helper, token_start, token_end, replace_token)

        tokens[i] = replace_token + tokens[i].whitespace_

    handle_whitespaces = lambda token: token if type(token) == str else token.text_with_ws
    new_context = nlp("".join([handle_whitespaces(token) for token in tokens]))
    document['context'] = new_context.text

    update_answer_positions(document)



def mask_documents(flt):
    # full processing path
    test_file = read_test_file(TEST_FILENAME)
    backup_test_file(test_file)

    if not test_file:
        print('Test file empty. Nothing to do')
        return

    datas = test_file['data']

    for data in datas:
        documents = data['paragraphs']
        for document in documents:
            mask_document(document, flt)
            # test_file['data']['paragraphs'][i] = filtered_document

    write_test_file(test_file, TEST_FILENAME)


def remove_cache_files():
    cache_path = './cached_dev_roberta-base_512'
    if os.path.exists(cache_path):
        os.remove(cache_path)


# def eval_model(common_plots):
#     return None, None


def run_and_eval():
    remove_cache_files()

    advanced_scores = run_model(MODEL_NAME, PATH_TO_VENV)
    prec_recall_scores, precisions, recalls = stats_precisions_recalls(MODEL_NAME)

    return {
        'advanced_scores': advanced_scores,
        'prec_recall_scores': prec_recall_scores,
        'precisions': precisions,
        'recalls': recalls
    }


def process_results(global_stats, common_plots):
    pass


def process_filter(flt):
    # phase 1: mask documents
    mask_documents(flt)
    
    # phase 2: predict masked documents with model
    # don't forget about cache files
    stats = run_and_eval()

    # phase 3: unmask documents
    unmask_documents(flt)

    return stats


def average_stats(stats):
    advanced_scores = []
    prec_recall_scores = []
    precisions = []
    recalls = []

    for stat in stats:
        advanced_scores.append(stat['advanced_scores'])
        prec_recall_scores.append(stat['prec_recall_scores'])
        precisions.append(stat['precisions'])
        recalls.append(stat['recalls'])

    average_advanced_scores = [sum(els)/len(advanced_scores) for els in zip(*advanced_scores)]
    average_prec_recall_scores = [sum(els)/len(prec_recall_scores) for els in zip(*prec_recall_scores)]
    average_precisions = [sum(els)/len(precisions) for els in zip(*precisions)]
    average_recalls = [sum(els)/len(recalls) for els in zip(*recalls)]


    averaged_stats = {
        'advanced_scores': average_advanced_scores,
        'prec_recall_scores': average_prec_recall_scores,
        'precisions': average_precisions,
        'recalls': average_recalls
    }

    return averaged_stats


def save_stats_to_file(new_stats, flt_name, filename=STATS_FILENAME):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            stats = json.load(file)
            stats[flt_name] = new_stats
    else:
        stats = {
            flt_name: new_stats
        }
    with open(filename, 'w') as file:
        json.dump(stats, file)



def main():
    global_stats = []
    common_plots = {}

    for flt in filters:
        stats = None
        if flt['num_runs']:
            stats_to_be_averaged = []
            # plots_to_be_averaged = []

            for _ in range(flt['num_runs']):
                stats = process_filter(flt) # plots
                stats_to_be_averaged.append(stats)
                # plots_to_be_averaged.append(plots)
            stats = average_stats(stats_to_be_averaged) #, plots_to_be_averaged)
        else:
            # stats, plots, common_plots = process_filter(flt, common_plots)
            stats = process_filter(flt)
        

        save_stats_to_file(stats, flt['name'])
        # phase x: evaluate model performance, create plots, etc.
        # mutates common plots
        eval_model(common_plots, stats['recalls'], stats['precisions'], stats['prec_recall_scores'], flt['name'])


        global_stats.append({
            'name': flt['name'],
            'stats': stats,
            # 'plots': plots
        })

    # process_results(global_stats, common_plots)
    # remove cached files


if __name__ == "__main__":
    main()
    






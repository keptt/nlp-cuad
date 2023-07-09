import json
from transformers import AutoTokenizer
import spacy
import random
import os
import os.path as osp
from evaluate_model import run_model, stats_precisions_recalls
from collections import OrderedDict
import numpy as np
import math


is_chance = lambda chance: random.random() <= chance


filters = [
    {
        "name": "Baseline",
        "filtered_tokens": lambda doc: [],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": None,
    },
    {  # probabilistic filter
        "name": "MaskProperNounsP(0.25)",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent if is_chance(0.25)],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": 5,
    },
    {  # probabilistic filter
        "name": "MaskProperNounsP(0.5)",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent if is_chance(0.5)],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": 5,
    },
    {  # probabilistic filter
        "name": "MaskProperNounsP(0.75)",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent if is_chance(0.75)],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": 5,
    },
    {
        "name": "MaskAllProperNouns",
        "filtered_tokens": lambda doc: [token.i for ent in doc.ents for token in ent],
        "condition": lambda ent, doc: True,
        "mask_with": None,
        "num_runs": None,
    },
]

TEST_FILENAME = "./data/test.json"
TEST_FILENAME_TEMPLATE = "./data/test{}.json"
TEST_BACKUP_FILENAME = "./data/test.bak.json"
MODEL_NAME = 'roberta-base'
PATH_TO_PYTHON_PROJECT = os.getcwd() # osp.join(os.getcwd(), osp.join('..', '..'))
PATH_TO_VENV = os.path.join(PATH_TO_PYTHON_PROJECT, 'env', 'Scripts', 'python')

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



def take_min_stats(list_of_stats):
    advanced_scores = []
    prec_recall_scores = []
    precisions = []
    recalls = []

    for stat in list_of_stats:
        advanced_scores.append(stat['advanced_scores'])
        prec_recall_scores.append(stat['prec_recall_scores'])
        precisions.append(stat['precisions'])
        recalls.append(stat['recalls'])

    # Initialize a new OrderedDict to hold the sums
    minimized_advanced_scores = OrderedDict.fromkeys(advanced_scores[0], math.inf)

    # Add up the values from each OrderedDict
    for d in advanced_scores:
        for key, value in d.items():
            if value < minimized_advanced_scores[key]:
                minimized_advanced_scores[key] = value

    minimized_prec_recall_scores = OrderedDict.fromkeys(prec_recall_scores[0], math.inf)
    del(minimized_prec_recall_scores['name'])

    # Add up the values from each OrderedDict
    for d in prec_recall_scores:
        for key, value in d.items():
            if key == 'name':
                pass
            else:
                if value < minimized_advanced_scores[key]:
                    minimized_prec_recall_scores[key] = value

    minimized_precisions = np.array(precisions).min(axis=0)
    minimized_recalls = np.array(recalls).min(axis=0)

    minimized_stats = {
        'advanced_scores': minimized_advanced_scores,
        'prec_recall_scores': minimized_prec_recall_scores,
        'precisions': list(minimized_precisions),
        'recalls': list(minimized_recalls)
    }

    return minimized_stats



def take_max_stats(list_of_stats):
    advanced_scores = []
    prec_recall_scores = []
    precisions = []
    recalls = []

    for stat in list_of_stats:
        advanced_scores.append(stat['advanced_scores'])
        prec_recall_scores.append(stat['prec_recall_scores'])
        precisions.append(stat['precisions'])
        recalls.append(stat['recalls'])

    # Initialize a new OrderedDict to hold the sums
    maximized_advanced_scores = OrderedDict.fromkeys(advanced_scores[0], -math.inf)

    # Add up the values from each OrderedDict
    for d in advanced_scores:
        for key, value in d.items():
            if value > maximized_advanced_scores[key]:
                maximized_advanced_scores[key] = value

    maximized_prec_recall_scores = OrderedDict.fromkeys(prec_recall_scores[0], -math.inf)
    del(maximized_prec_recall_scores['name'])

    # Compare the values from each OrderedDict
    for d in prec_recall_scores:
        for key, value in d.items():
            if key == 'name':
                pass
            else:
                if value > maximized_advanced_scores[key]:
                    maximized_prec_recall_scores[key] = value

    maximized_precisions = np.array(precisions).max(axis=0)
    maximized_recalls = np.array(recalls).max(axis=0)

    maximized_stats = {
        'advanced_scores': maximized_advanced_scores,
        'prec_recall_scores': maximized_prec_recall_scores,
        'precisions': list(maximized_precisions),
        'recalls': list(maximized_recalls)
    }

    return maximized_stats



def average_stats(list_of_stats):
    advanced_scores = []
    prec_recall_scores = []
    precisions = []
    recalls = []

    for stat in list_of_stats:
        advanced_scores.append(stat['advanced_scores'])
        prec_recall_scores.append(stat['prec_recall_scores'])
        precisions.append(stat['precisions'])
        recalls.append(stat['recalls'])

    # Initialize a new OrderedDict to hold the sums
    minimize_advanced_scores = OrderedDict.fromkeys(advanced_scores[0], 0)

    # Add up the values from each OrderedDict
    for d in advanced_scores:
        for key, value in d.items():
            minimize_advanced_scores[key] += value

    average_prec_recall_scores = OrderedDict.fromkeys(prec_recall_scores[0], 0)
    del(average_prec_recall_scores['name'])

    # Compare the values from each OrderedDict
    for d in prec_recall_scores:
        for key, value in d.items():
            if key == 'name':
                pass
            else:
                average_prec_recall_scores[key] += value

    # Divide by the number of OrderedDicts to get the averages
    minimize_advanced_scores = dict((key, value / len(advanced_scores)) for key, value in minimize_advanced_scores.items())
    average_precisions = np.array(precisions).mean(axis=0)
    average_recalls = np.array(recalls).mean(axis=0)
    average_prec_recall_scores = dict((key, value / len(prec_recall_scores)) for key, value in average_prec_recall_scores.items())
    # average_prec_recall_scores = [sum(els)/len(prec_recall_scores) for els in zip(*prec_recall_scores)]
    # average_precisions = [sum(els)/len(precisions) for els in zip(*precisions)]
    # average_recalls = [sum(els)/len(recalls) for els in zip(*recalls)]

    averaged_stats = {
        'advanced_scores': minimize_advanced_scores,
        'prec_recall_scores': average_prec_recall_scores,
        'precisions': list(average_precisions),
        'recalls': list(average_recalls)
    }

    return averaged_stats


def save_stats_to_file(new_stats, flt_name, filename='stats.json'):
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



def process_stats_for_probabilistic_filters(list_of_stats):
    avg_stats = average_stats(list_of_stats) #, plots_to_be_averaged)
    min_stats = take_min_stats(list_of_stats) #, plots_to_be_averaged)
    max_stats = take_max_stats(list_of_stats) #, plots_to_be_averaged)

    return avg_stats, min_stats, max_stats



def main():
    # global_stats = []
    common_plots = {}
    stats = []
    min_stats = None
    max_stats = None

    for flt in filters:
        stats = None
        min_stats = None
        max_stats = None

        if flt['num_runs']:
            list_of_stats = []
            # plots_to_be_averaged = []

            for _ in range(flt['num_runs']):
                stats = process_filter(flt) # plots
                list_of_stats.append(stats)
                # plots_to_be_averaged.append(plots)
            stats, min_stats, max_stats = process_stats_for_probabilistic_filters(list_of_stats)

        else:
            # stats, plots, common_plots = process_filter(flt, common_plots)
            stats = process_filter(flt)
        
        # phase x: evaluate model performance, create plots, etc.
        # mutates common plots
        save_stats_to_file(stats, flt['name'], 'stats.json')
        if min_stats and max_stats:
            save_stats_to_file(min_stats, flt['name'], 'stats.min.json')
            save_stats_to_file(max_stats, flt['name'], 'stats.max.json')
        else:
            save_stats_to_file(stats, flt['name'], 'stats.min.json')
            save_stats_to_file(stats, flt['name'], 'stats.max.json')

        # eval_model(common_plots, stats['recalls'], stats['precisions'], stats['prec_recall_scores'], flt['name'])
        #
        # global_stats.append({
        #     'name': flt['name'],
        #     'stats': stats,
        #     # 'plots': plots
        # })

    # process_results(global_stats, common_plots)
    # remove cached files


if __name__ == "__main__":
    main()
 




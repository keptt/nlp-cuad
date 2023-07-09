import subprocess
import shlex
import os
import os.path as osp
# from evaluate import stats_precisions_recalls
from collections import OrderedDict
from evaluate import get_results, load_json, get_answers

# def remove_cache():
#     cache_path = osp.join(PATH_TO_PYTHON_PROJECT, osp.join(osp.join('project', 'CUAD_v1'), 'cached_dev_roberta-base_512'))
#     if osp.exists(cache_path):
#         os.remove(cache_path)


def stats_precisions_recalls(model_type):
    test_json_path = "./data/test.json"
    model_path = f"./trained_models/{model_type}"

    gt_dict = load_json(test_json_path)
    gt_dict = get_answers(gt_dict)

    final_stats, precisions, recalls = get_results(model_path, gt_dict, verbose=True)
    return final_stats, precisions, recalls


def run_model(model_type, venv_path):
    # Assuming you're running this code from the 'CUAD_v1' directory
    args = [venv_path,
            "train.py"
            ]
    
    #! settings for running on an instance with high end GPU
    # args.extend(shlex.split(f"--output_dir ./trained_models/{model_type} --model_type {model_type.split('-')[0]} "
    #                         f"--model_name_or_path ./trained_models/{model_type} --predict_file ./data/test.json "
    #                         "--do_eval "
    #                         "--version_2_with_negative  --per_gpu_eval_batch_size=800 --max_seq_length 512 "
    #                         "--threads 30 "
    #                         "--max_answer_length "
    #                         "512 --doc_stride 256 --save_steps 1000 --n_best_size 20 --overwrite_output_dir"))
    # process = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)


    args.extend(shlex.split(f"--output_dir ./trained_models/{model_type} --model_type {model_type.split('-')[0]} "
                            f"--model_name_or_path ./trained_models/{model_type} --predict_file ./data/test.json "
                            "--do_eval "
                            "--version_2_with_negative  --per_gpu_eval_batch_size=40 --max_seq_length 512 "
                            "--max_answer_length "
                            "512 --doc_stride 256 --save_steps 1000 --n_best_size 20 --overwrite_output_dir"))
    process = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    stdout = process.communicate()
    string_with_stats = stdout[0][stdout[0].find("OrderedDict"):stdout[0].find("")]
    string_with_stats = string_with_stats.encode('utf-8')
    string_with_stats = string_with_stats.decode('utf-8')
    results = eval(string_with_stats)
    return results


if __name__ == "__main__":
    model_type = 'roberta'
    # remove_cache()
    results = run_model(model_type)
    stats, precisions, recalls = stats_precisions_recalls(model_type)
    print("----------------------------------")

import matplotlib.pyplot as plt
import seaborn as sns
import evaluate
import numpy as np


import pandas as pd
import os
import json

NUM_TERMS = 3
FILTER_SMOOTH = np.array([1. / NUM_TERMS] * NUM_TERMS)

##This is called once per filter
def eval_model(common_plots,recalls, precisions, results, filter_name, smoothing=True):
    if smoothing:
        precisions, recalls = np.array(precisions), np.array(recalls)
        padding_len = (len(FILTER_SMOOTH) - 1) // 2
        precisions = np.concatenate([precisions[:padding_len], precisions, precisions[-padding_len:]])
        precisions = np.convolve(np.array(precisions), FILTER_SMOOTH, mode='valid')
        recalls = np.concatenate([recalls[:padding_len], recalls, recalls[-padding_len:]])
        recalls = np.convolve(np.array(recalls), FILTER_SMOOTH, mode='valid')
    plot_precision_recall_curve(common_plots,recalls, precisions, filter_name)
    fig_precision_recall_curve = common_plots.get("prc")[0]
    fig_precision_recall_curve.savefig('./precision_recall_curve.png')

    plot_barplot(common_plots,filter_name,"Precission at 80 recall",results.get("prec_at_80_recall"))
    fig_barplot_prec_at_80_recall = common_plots.get("Precission at 80 recall")[0]
    fig_barplot_prec_at_80_recall.savefig('./barplot_prec_at_80_recall.png')

    plot_barplot(common_plots,filter_name,"Precission at 90 recall",results.get("prec_at_90_recall"))
    fig_barplot_prec_at_90_recall = common_plots.get("Precission at 90 recall")[0]
    fig_barplot_prec_at_90_recall.savefig('./barplot_prec_at_90_recall.png')

    plot_barplot(common_plots,filter_name,"Area under precission curve",results.get("aupr"))
    fig_barplot_aupr = common_plots.get("Area under precission curve")[0]
    fig_barplot_aupr.savefig('./barplot_aupr.png')

    # plot_barplot_aupr_by_category("1","2",common_plots)


def plot_from_file(common_plots,filename, smoothing=True):
    stats=read_stats_from_file(filename)
    for filter in stats.keys():
        precisions=stats.get(filter).get("precisions")
        recalls=stats.get(filter).get("recalls")
        prec_recall_scores=stats.get(filter).get("prec_recall_scores")
        eval_model(common_plots,precisions,recalls,prec_recall_scores,filter, smoothing=smoothing)

def read_stats_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            stats = json.load(file)
        return stats
    else:
        print("File does not exist.")
        return None


def plot_precision_recall_curve(common_plots,recalls, precisions, filter_name):
    
    ##check if common_plots is empty
    if not "prc" in common_plots.keys():
        fig,ax = plt.subplots(figsize=(10,8))
        common_plots["prc"]=(fig,ax)
    
    fig,ax=common_plots.get("prc")

    sns.set_style("whitegrid")
    sns.lineplot(x=recalls, y=precisions, marker='o',ax=ax, label=filter_name)
    ax.set_title('Precision-Recall curve', fontsize=20)
    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)
    ax.legend(loc='best')

    common_plots["prc"] = (fig,ax)
    return

def read_test_json(test_json_path="./data/test_masked.json", model_path = "./trained_models/roberta-base", save_dir = "./results"):


    if not os.path.exists(save_dir): os.mkdir(save_dir)

    with open(test_json_path, "r") as f:
        gt_dict = json.load(f)
    gt_dict = evaluate.get_answers(gt_dict)


    predictions_path = os.path.join(model_path, "nbest_predictions_.json")
    name = model_path.split("/")[-1]

    with open(predictions_path, "r") as f:
        pred_dict = json.load(f)

    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))
    return pred_dict, gt_dict


##data_name specifies which data is to be plotted, e.g. prec_at_80_recall, aup,...
def plot_barplot(common_plots,x_name, data_name, data):
    ##check if common_plots is empty
    if not data_name in common_plots.keys():
        fig,ax = plt.subplots()
        common_plots[data_name]=(fig,ax)
    
    fig,ax=common_plots.get(data_name)


    #sns.barplot(x=[filter_name], y=[prec_at_80_recall], ax=ax)
    ax.barh(x_name,data)

    # Set labels and title
    ax.set_title(data_name)
    ax.set_xlabel('Filter')
    ax.set_ylabel(data_name)

    common_plots[data_name] = (fig,ax)
    return

##This barplot gets filtered auprs and unfiltered 
##auprs and compares both by drawing the filtered bars over the unfiltered in a darker color
##compare https://pythonbasics.org/seaborn-barplot/
## filtered_data,unfiltered_data are dicts

def plot_barplot_with_total(filtered_data,unfiltered_data):

    fig,ax = plt.subplots()


    #unfiltered_df = pd.DataFrame(list(unfiltered_data.items()), columns=["category", "y_unfiltered"])
    #filtered_df = pd.DataFrame(list(filtered_data.items()), columns=["category", "y_filtered"])
    #merged_df = pd.concat([unfiltered_df, filtered_df], axis=1)
    #merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]

    df = pd.DataFrame({
        'category': list(filtered_data.keys()),
        'AUPR with filter': list(filtered_data.values()),
        'AUPR without filter': list(unfiltered_data.values())
    })

    sns.set_context('paper')
    sns.set_color_codes('muted')

    ## x and y values are "swapped" because we want horizontal bars (growing from left to right)
    sns.barplot(y = "category", x = "AUPR without filter", data=df, label = 'Total', color = 'r', edgecolor = 'w', ax=ax)
    sns.set_color_codes('pastel')

    sns.barplot(y = "category", x = "AUPR with filter", data=df, label = 'Filtered', color = 'b', edgecolor = 'w',ax=ax)
    ax.set_xlabel("AUPR")
    ax.legend(ncol = 2, loc = 'lower right')
    sns.despine(left = True, bottom = True)

    fig.savefig('aupr_with_total.png')
    return fig,ax

def test_plot_barplot_with_total():
    pred_dict, gt_dict = read_test_json()

    categories=['Document Name', 'Parties', 'Agreement Date', 'Effective Date', 'Expiration Date', 'Renewal Term', 'Notice Period to Terminate Renewal', 'Governing Law', 'Most Favored Nation', 'Non-Compete', 'Exclusivity', 'No-Solicit of Customers', 'Competitive Restriction Exception', 'No-Solicit of Employees', 'Non-Disparagement', 'Termination for Convenience', 'Rofr/Rofo/Rofn', 'Change of Control', 'Anti-Assignment', 'Revenue/Profit Sharing', 'Price Restrictions', 'Minimum Commitment', 'Volume Restriction', 'IP Ownership Assignment', 'Joint IP Ownership', 'License Grant', 'Non-Transferable License', 'Affiliate License-Licensor', 'Affiliate License-Licensee', 'Unlimited/All-You-Can-Eat-License', 'Irrevocable or Perpetual License', 'Source Code Escrow', 'Post-Termination Services', 'Audit Rights', 'Uncapped Liability', 'Cap on Liability', 'Liquidated Damages', 'Warranty Duration', 'Insurance', 'Covenant Not to Sue', 'Third Party Beneficiary']
    
    aupr_dict=dict()
    for category in categories:
        precisions, recalls, confs = evaluate.get_precisions_recalls(pred_dict, gt_dict, category=category)
        aupr = evaluate.get_aupr(precisions, recalls)
        aupr_dict[category]=aupr
        #plot_barplot(category,"Area under precission by category", aupr)
        


    test_filtered_aupr_dict = dict(aupr_dict)
    for key in test_filtered_aupr_dict.keys():
        test_filtered_aupr_dict[key]= test_filtered_aupr_dict.get(key) + 5


    ##swap roles of args
    fig,ax = plot_barplot_with_total(aupr_dict,test_filtered_aupr_dict)
    fig.savefig('test_aupr_with_total.png')


def get_aupr_by_category(test_json_path, model_path = "./trained_models/roberta-base", save_dir = "./results"):
    
    ##Read data
    ## by default from test_json_path="./data/test_masked.json", model_path = "./trained_models/roberta-base", save_dir = "./results"
    pred_dict, gt_dict = read_test_json()

    categories=['Document Name', 'Parties', 'Agreement Date', 'Effective Date', 'Expiration Date', 'Renewal Term', 'Notice Period to Terminate Renewal', 'Governing Law', 'Most Favored Nation', 'Non-Compete', 'Exclusivity', 'No-Solicit of Customers', 'Competitive Restriction Exception', 'No-Solicit of Employees', 'Non-Disparagement', 'Termination for Convenience', 'Rofr/Rofo/Rofn', 'Change of Control', 'Anti-Assignment', 'Revenue/Profit Sharing', 'Price Restrictions', 'Minimum Commitment', 'Volume Restriction', 'IP Ownership Assignment', 'Joint IP Ownership', 'License Grant', 'Non-Transferable License', 'Affiliate License-Licensor', 'Affiliate License-Licensee', 'Unlimited/All-You-Can-Eat-License', 'Irrevocable or Perpetual License', 'Source Code Escrow', 'Post-Termination Services', 'Audit Rights', 'Uncapped Liability', 'Cap on Liability', 'Liquidated Damages', 'Warranty Duration', 'Insurance', 'Covenant Not to Sue', 'Third Party Beneficiary']
    
    aupr_dict=dict()
    for category in categories:
        precisions, recalls, confs = evaluate.get_precisions_recalls(pred_dict, gt_dict, category=category)
        aupr = evaluate.get_aupr(precisions, recalls)
        aupr_dict[category]=aupr
    
    return aupr_dict


def plot_barplot_aupr_by_category(common_plots, filter_name):
    


    aupr_dict_no_filter = get_aupr_by_category(test_json_path="./data/test_masked.json")

    ##TODO: check naming convention for filter test json names
    aupr_dict_some_filter = get_aupr_by_category(test_json_path="./data/test_"+ filter_name +"_masked.json") 

    fig,ax = plot_barplot_with_total(aupr_dict_some_filter,aupr_dict_no_filter)

    ax.set_title("aupr with filter " + filter_name)
    
    common_plots["aupr_by_category_filter_" + filter_name] = (fig,ax)



if __name__ =='__main__':
    common_plots = {}
    plot_from_file(common_plots, 'stats.json', smoothing=True)
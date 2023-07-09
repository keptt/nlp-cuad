
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate


##temporary
import pandas as pd
import os
import json


##This is called once per filter
def eval_model(common_plots, recalls, precisions, results, filter_name):
    plot_precision_recall_curve(common_plots,recalls, precisions)
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

    plot_barplot_aupr_by_category("1","2",common_plots)

def plot_precision_recall_curve(common_plots,recalls, precisions):
    
    ##check if common_plots is empty
    if not "prc" in common_plots.keys():
        fig,ax = plt.subplots(figsize=(10,8))
        common_plots["prc"]=(fig,ax)
    
    fig,ax=common_plots.get("prc")

    sns.set_style("whitegrid")
    sns.lineplot(x=recalls, y=precisions, marker='o',ax=ax)
    ax.set_title('Precision-Recall curve', fontsize=20)
    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)

    common_plots["prc"] = (fig,ax)
    return


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

def plot_barplot_with_total(common_plots,filtered_data,unfiltered_data):

    fig,ax = plt.subplots()

    unfiltered_df = pd.DataFrame(list(unfiltered_data.items()), columns=["x_column_name", "y_column_name"])
    sns.set_context('paper')
    sns.set_color_codes('muted')

    ## x and y values are "swapped" because we want horizontal bars (growing from left to right)
    sns.barplot(y = "x_column_name", x = "y_column_name", data=unfiltered_df, label = 'Total', color = 'b', edgecolor = 'w')
    sns.set_color_codes('pastel')


    filtered_df = pd.DataFrame(list(filtered_data.items()), columns=["x_column_name", "y_column_name"])


    sns.barplot(y = "x_column_name", x = "y_column_name", data=filtered_df, label = 'Alcohol-involved', color = 'b', edgecolor = 'w')
    ax.legend(ncol = 2, loc = 'lower right')
    sns.despine(left = True, bottom = True)


def plot_barplot_aupr_by_category(model_path, gt_dict, common_plots):
    
    ###BEGIN TESTING###
    #Manually read json
    test_json_path = "./data/test_masked.json"
    model_path = "./trained_models/roberta-base"
    save_dir = "./results"
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    with open(test_json_path, "r") as f:
        gt_dict = json.load(f)
    gt_dict = evaluate.get_answers(gt_dict)
    ###END TESTING###


    predictions_path = os.path.join(model_path, "nbest_predictions_.json")
    name = model_path.split("/")[-1]

    with open(predictions_path, "r") as f:
        pred_dict = json.load(f)

    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))

    categories=['Document Name', 'Parties', 'Agreement Date', 'Effective Date', 'Expiration Date', 'Renewal Term', 'Notice Period to Terminate Renewal', 'Governing Law', 'Most Favored Nation', 'Non-Compete', 'Exclusivity', 'No-Solicit of Customers', 'Competitive Restriction Exception', 'No-Solicit of Employees', 'Non-Disparagement', 'Termination for Convenience', 'Rofr/Rofo/Rofn', 'Change of Control', 'Anti-Assignment', 'Revenue/Profit Sharing', 'Price Restrictions', 'Minimum Commitment', 'Volume Restriction', 'IP Ownership Assignment', 'Joint IP Ownership', 'License Grant', 'Non-Transferable License', 'Affiliate License-Licensor', 'Affiliate License-Licensee', 'Unlimited/All-You-Can-Eat-License', 'Irrevocable or Perpetual License', 'Source Code Escrow', 'Post-Termination Services', 'Audit Rights', 'Uncapped Liability', 'Cap on Liability', 'Liquidated Damages', 'Warranty Duration', 'Insurance', 'Covenant Not to Sue', 'Third Party Beneficiary']
    
    ##To print out a list of all categories:
    #df = pd.read_csv("category_descriptions.csv")
    #q_dict = {}
    #for i in range(df.shape[0]):
    #    category = df.iloc[i, 0].split("Category: ")[1]
    #    print(category)
    #    categories.append(category)
    #print(categories)

    aupr_dict=dict()
    for category in categories:
        precisions, recalls, confs = evaluate.get_precisions_recalls(pred_dict, gt_dict, category=category)
        aupr = evaluate.get_aupr(precisions, recalls)
        aupr_dict[category]=aupr
        #plot_barplot(common_plots,category,"Area under precission by category", aupr)
        
    #fig,ax = plt.subplots()
    #df = pd.DataFrame(list(aupr_dict.items()), columns=['category', 'aupr'])
    #sns.set_context('paper')
    #sns.set_color_codes('pastel')
    #sns.barplot(y = "category", x = "aupr", data=df, label = 'Total', color = 'b', edgecolor = 'w')
    #sns.set_color_codes('muted')
    ##For testing purposes, just use the original aupr data * 0.5 as the filtered data
    #df_filtered=df
    #df_filtered["aupr"]=df_filtered["aupr"]*0.5
    #sns.barplot(y = "category", x = "aupr", data=df_filtered, label = 'Alcohol-involved', color = 'b', edgecolor = 'w')
    #ax.legend(ncol = 2, loc = 'lower right')
    #sns.despine(left = True, bottom = True)

    test_filtered_aupr_dict = aupr_dict
    for key in test_filtered_aupr_dict.keys():
        test_filtered_aupr_dict[key]= 0.5 * test_filtered_aupr_dict.get(key)

    plot_barplot_with_total(common_plots,test_filtered_aupr_dict,aupr_dict)
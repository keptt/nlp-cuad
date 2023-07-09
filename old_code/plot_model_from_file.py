def plot_from_file(common_plots,filename):
    stats=read_stats_from_file(filename)
    for filter in stats.keys():
        precisions=stats.get(filter).get("precisions")
        recalls=stats.get(filter).get("recalls")
        prec_recall_scores=stats.get(filter).get("prec_recall_scores")
        eval_model(common_plots,precisions,recalls,prec_recall_scores,filter)

def read_stats_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            stats = json.load(file)
        return stats
    else:
        print("File does not exist.")
        return None
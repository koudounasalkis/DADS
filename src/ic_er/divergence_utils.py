from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 

from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence

import warnings
warnings.filterwarnings("ignore")

"""
    Function for discretizing the dataframe
    @input:
        - df: dataframe to discretize
        - bins: number of bins
        - attributes: columns to discretize
        - strategy: strategy for discretization
        - adaptive: flag specifying whether to use adaptive discretization
        - round_v: number of decimals to round
        - min_distinct: minimum number of distinct values to discretize
    @output:
        - df_discretized: dataframe discretized
"""
def discretize(
    dfI,
    bins=3,
    attributes=None,
    strategy="quantile",
    adaptive=True,
    round_v=0,
    min_distinct=10
    ):
    attributes = dfI.columns if attributes is None else attributes
    
    X_discretized = KBinsDiscretizer_continuos(
        dfI,
        attributes,
        bins=bins,
        strategy=strategy,
        adaptive=adaptive,
        round_v=round_v,
        min_distinct=min_distinct,
    )
    for attribute in dfI.columns:
        if attribute not in X_discretized:
            X_discretized[attribute] = dfI[attribute]
    return X_discretized

def KBinsDiscretizer_continuos(
    dt, 
    attributes=None, 
    bins=3, 
    strategy="quantile", 
    adaptive=True, 
    round_v=0, 
    min_distinct=10
    ):
    
    def _get_edges(input_col, bins, round_v=0):
        from sklearn.preprocessing import KBinsDiscretizer

        est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
        est.fit(input_col)
        edges = [i for i in est.bin_edges_][0]
        edges = [round(i, round_v) for i in edges][1:-1]

        if len(set(edges)) != len(edges):
            edges = [
                edges[i]
                for i in range(0, len(edges))
                if len(edges) - 1 == i or edges[i] != edges[i + 1]
            ]
        return edges

    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != object]
    X_discretize = dt[attributes].copy()
    for col in continuous_attributes:
        if len(dt[col].value_counts()) > min_distinct:

            if adaptive:
                msg = None
                found = False
                for increased in range(0, 5):
                    edges = _get_edges(dt[[col]], bins + increased, round_v=round_v)
                    if (len(edges) + 1) != bins:
                        msg = f"Not enough data in the bins for attribute {col}\
                            --> bin size is increased from {bins} to {bins+increased}"
                    else:
                        found = True
                        break
                if found == False:
                    edges = _get_edges(dt[[col]], bins, round_v=round_v)
                    msg = f"Not enough data in the bins & adaptive failed for attribute {col}. \
                        Discretized with lower #of bins ({len(edges)} vs {bins})"
                if msg:
                    warnings.warn(msg)
            else:
                edges = _get_edges(dt[[col]], bins, round_v=round_v)

            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]:.{round_v}f}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]:.{round_v}f}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[
                    data_idx, col
                ] = f"({edges[i-1]:.{round_v}f}-{edges[i]:.{round_v}f}]"
            ### IMPO: added check if no discretization is performed.
            # In this case, the attribute is dropped.
            if edges == []:
                msg = f"No discretization is performed for attribute '{col}'. \
                    The attribute {col} is removed. \n\
                    Consider changing the size of the bins or the strategy.'"
                warnings.warn(msg)
                X_discretize.drop(columns=[col], inplace=True)
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    return X_discretize

""" 
    Function for discretizing the dataframe
    @input: 
        - df: dataframe to discretize
        - target_col: prediction column
        - train: flag specifying whether it is a train or test df
    @output: 
        - df_discretized: dataframe discretized
"""
def discretize_df(
    df, 
    target_col, 
    dataset, 
    approach, 
    train=False, 
    verbose=False
    ): 

    ## Target to be used for discretization
    if train:
        target_col = 'path'
        if verbose:
            print("Discretizing training data...")
    else:
        target_col = 'prediction'
        if verbose:
            print("Discretizing test data...")
        
    if approach == 'divexplorer':
        ## Columns to be used for discretization
        if dataset == 'fsc':
            demo_cols = ['Self-reported fluency level ', 'First Language spoken',
                'Current language used for work/school', 'gender', 'ageRange']
            slot_cols = ['action', 'object', 'location']
            signal_cols = ['total_silence', 'total_duration', 'trimmed_duration', 
                'n_words', 'speed_rate_word', 'speed_rate_word_trimmed'] 
        elif dataset == 'italic':
            demo_cols = ['gender', 'age', 'region', 'nationality', 'lisp', 'education']
            slot_cols = ['environment', 'device', 'field', 'intent']
            signal_cols = ['total_silence', 'total_duration', 'trimmed_duration', 
                'n_words', 'speed_rate_word', 'speed_rate_word_trimmed'] 
        elif dataset == 'iemocap':
            demo_cols = ['gender']
            slot_cols = ['emotion']
            signal_cols = ['valence', 'activation', 'dominance', 
                'total_silence', 'total_duration', 'trimmed_duration',
                'n_words', 'speed_rate_word', 'speed_rate_word_trimmed']
        elif dataset == 'librispeech':
            demo_cols = ['gender']
            signal_cols = ['total_silence', 'total_duration', 'trimmed_duration', 
                'n_words', 'n_pauses', 'speed_rate_word', 'speed_rate_word_trimmed'] 
            slot_cols = []

        input_cols = demo_cols + slot_cols + signal_cols
            
        ## Discretize dataframe
        df_discretized = discretize(
            df[input_cols+[target_col]],
            bins=3,
            attributes=input_cols,
            strategy="quantile",    
            round_v = 2,
            min_distinct=5
            )

        ## Replace values with ranges: "low", "medium", "high"
        replace_values = {}
        for i in range(0,len(signal_cols)):
            for v in df_discretized[signal_cols[i]].unique():
                if "<=" == v[0:2]:
                    replace_values[v] = "low"
                elif ">" == v[0]:
                    replace_values[v] = "high"
                elif "("  == v[0] and "]"  == v[-1]:
                    replace_values[v] = "medium"
                else:
                    raise ValueError(v)
            df_discretized[signal_cols[i]].replace(replace_values, inplace=True)
        
        if dataset == 'fsc':
            df_discretized.loc[df_discretized["location"]=="none_location", "location"] = "none"
            df_discretized.loc[df_discretized["object"]=="none_object", "object"] = "none"

    elif approach == 'clustering':
        ## Columns to be used (just the cluster ids)
        input_cols = [f'speech_cluster_id_{k}' for k in [10,20,50]]
        if dataset == 'fsc' or dataset == 'italic':
            df_discretized = df[input_cols+[target_col]+['intent']]
        elif dataset == 'iemocap':
            df_discretized = df[input_cols+[target_col]+['emotion']]
        elif dataset == 'librispeech':
            df_discretized = df[input_cols+[target_col]]

    elif approach == 'knn':
        input_cols = [f'knn_id_{k}' for k in range(2,20)]
        if dataset == 'fsc' or dataset == 'italic':
            df_discretized = df[input_cols+[target_col]+['intent']]
        elif dataset == 'iemocap':
            df_discretized = df[input_cols+[target_col]+['emotion']]
        elif dataset == 'librispeech':
            df_discretized = df[input_cols+[target_col]]

    else:
        raise ValueError(f"Approach {approach} not supported")

    if verbose:
        print("Data discretized!\n")
    return df_discretized
    

""" 
    Function for compute the divergence
    @input: 
        - df_discretized: dataframe discretized
        - target_col: prediction column
        - target_div: target divergence
        - target_metric: target metric
        - approach: approach used for discretization
        - min_sup: minimum support for frequent pattern mining
    @output:
        - FP_div: frequent pattern divergence
"""
def compute_divergence(
    df_discretized, 
    target_col, 
    target_div, 
    target_metric, 
    approach, 
    min_sup=0.03, 
    num_clusters=10, 
    verbose=False
    ):

    if verbose:
        print("Computing Divergence...")

    ## Divergence
    if approach == 'clustering':
        df_discretized_k = df_discretized[[f'speech_cluster_id_{k}' for k in [num_clusters]] + [target_col]]
        df_discretized = df_discretized_k.copy()
    elif approach == 'knn':
        df_discretized_k = df_discretized[[f'knn_id_{k}' for k in range(2,20)] + [target_col]]
        df_discretized = df_discretized_k.copy()

    fp_diver = FP_DivergenceExplorer(df_discretized, true_class_name=target_col, class_map={"P":1, "N":0})
    FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=[target_metric])

    ## Columns to be shown  
    show_cols = ['support', 'itemsets', '#errors', '#corrects', 'accuracy', \
                'd_accuracy', 't_value', 'support_count', 'length']
    remapped_cols = {'tn': '#errors', 'tp': '#corrects', 'posr': 'accuracy', \
                target_metric: target_div, 't_value_tp_fn': 't_value'}
    FP_fm.rename(columns=remapped_cols, inplace=True)
    FP_fm = FP_fm[show_cols].copy()

    ## Compute Divergence
    FP_fm['accuracy'] = round(FP_fm['accuracy'], 5)
    FP_fm['d_accuracy'] = round(FP_fm['d_accuracy'], 5)
    FP_fm['t_value'] = round(FP_fm['t_value'], 2)
    fp_divergence = FP_Divergence(FP_fm, target_div)
    FPdiv = fp_divergence.getDivergence(th_redundancy=None)[::-1]

    if verbose:
        print("Divergence computed!\n")

    return FPdiv



""" 
    Define Class Weights: this function computes the weights for each subgroup
    and assign them to the train data 
    @input:
        - folder: folder where the dataset is stored
        - df: dataframe
        - dataset: dataset name
        - approach: approach used for discretization
"""
def weights_rebalancing(
    folder, 
    df, 
    output_dir, 
    dataset, 
    approach, 
    min_sup=0.03, 
    num_clusters=10, 
    verbose=False
    ):

    if verbose:
        print(approach)
        print(dataset)

    if approach == 'random':
        df_discretized_train = pd.read_csv(os.path.join(folder, "train_data_80.csv"))
        df_discretized_train["weight"] = 1.0
        ## assign the weights randomly
        df_discretized_train.loc[df_discretized_train.index, "weight"] = np.random.uniform(
            0.0, 
            1.0, 
            len(df_discretized_train)
            )

    else:

        ## DivExplorer targets
        target_col = 'prediction' 
        target_metric = 'd_posr'
        target_div = 'd_accuracy'
        t_value_col = 't_value_tp_fn'

        ## Discretize the dataframe
        df_discretized = discretize_df(
            df=df, 
            target_col=target_col, 
            dataset=dataset, 
            approach=approach, 
            train=False,
            verbose=verbose
            )

        ## Compute Divergence
        FPdiv = compute_divergence(
            df_discretized, 
            target_col=target_col, 
            target_div=target_div, 
            target_metric=target_metric, 
            approach=approach,
            min_sup=min_sup,
            num_clusters=num_clusters,
            verbose=verbose
            )

        if verbose:
            print("The worst three subgroups are:")
            print("1: ", FPdiv.head(3).itemsets.values[0], "\
                with divergence: ", FPdiv.head(3)[target_div].values[0]*100, "%")
            print("2: ", FPdiv.head(3).itemsets.values[1], "\
                with divergence: ", FPdiv.head(3)[target_div].values[1]*100, "%")
            print("3: ", FPdiv.head(3).itemsets.values[2], "\
                with divergence: ", FPdiv.head(3)[target_div].values[2]*100, "%\n")
        with open(f"{output_dir}/worst_subgroups_{approach}.txt", "a") as f:
            f.write("The worst three subgroups are:\n")
            f.write("1: " + str(FPdiv.head(3).itemsets.values[0]) + "\
                with divergence: " + str(FPdiv.head(3)[target_div].values[0]*100) + "%\n")
            f.write("2: " + str(FPdiv.head(3).itemsets.values[1]) + "\
                with divergence: " + str(FPdiv.head(3)[target_div].values[1]*100) + "%\n")
            f.write("3: " + str(FPdiv.head(3).itemsets.values[2]) + "\
                with divergence: " + str(FPdiv.head(3)[target_div].values[2]*100) + "%\n\n")

        ## Compute class weights based on the Divergence, 
        # i.e., the lower the accuracy, the higher the weight of the corresponding samples
        if verbose:
            print("Rebalancing weights...\n")

        ## First, read and discretize the training data
        if approach == 'clustering' or approach == 'divexplorer' or approach == 'knn':
            df_train = pd.read_csv(os.path.join(folder, "train_data_80.csv"))
        else:
            raise ValueError("Approach not supported!")
        df_discretized_train = discretize_df(
            df_train, 
            target_col=target_col, 
            dataset=dataset, 
            approach=approach, 
            train=True
            )

        ## Now, for each retrieved subgroup, we add a new column in with the corresponding weight, 
        ## i.e., the higher the divergence the higher the weight,
        ## so that in the next training loop, we can use the weights to compute the weighted loss function
        df_discretized_train["weight"] = 1.0
        for i in tqdm(range(0, len(FPdiv))):
            sub = FPdiv.itemsets.values[i]
            d = list(sub)
            df_res = df_discretized_train.copy()
            for e in d:
                k, v = e.split("=")
                if approach == 'clustering':
                    df_res = df_res[df_res[k] == int(v)]
                elif approach == 'divexplorer':
                    df_res = df_res[df_res[k] == v]
                
                if len(df_res) > 0:
                    ## if df_discretized_train weight is 1.0, then we assign the weight, 
                    # if not we skip it, as it has already been assigned
                    if df_discretized_train.loc[df_res.index, "weight"].values[0] == 1.0:
                        df_discretized_train.loc[df_res.index, "weight"] = abs(FPdiv[target_div].values[i])
                    else:
                        continue

    if verbose:
        print("Weights computed!\n")

    ## Save the new training data with the weights
    df_discretized_train.to_csv(f"{folder}/new_data/train_data_80.csv", index=False)

    if verbose:
        print("New training data saved!\n")

        
        
import torch
import pandas as pd
import argparse
import numpy as np
import os
import random
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from ft_utils import WeightedTrainer, define_training_args, compute_metrics
from divergence_utils import weights_rebalancing
    
import warnings
warnings.filterwarnings("ignore")

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="number of training epochs",
        default=5,      # 5 for FSC and IEMOCAP, 7 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--steps",
        help="number of steps per epoch",
        default=500,    # 500 for FSC and IEMOCAP, 850 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,      # 1 for FSC and IEMOCAP, 4 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=500,    # 500 for FSC, 1000 for IEMOCAP, 5000 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--model",
        help="model to use for training",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_folder",
        help="path to the df folder",
        default="data/fsc",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results/ic/fsc",
        type=str,
        required=False)
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="fsc",
        type=str,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum duration of the audio files",
        default=4.0,   # 4.0 for FSC, 10.0 for IEMOCAP and ITALIC
        type=float,
        required=False)
    parser.add_argument(
        "--approach",
        help="approach to be used for bias mitigation",
        default="divexplorer",
        type=str,
        required=False)
    parser.add_argument(
        "--balancing",
        help="whether to use weights rebalancing",
        action="store_true",
        required=False)
    parser.add_argument(
        "--augmentation",
        help="whether to use data augmentation",
        action="store_true",
        required=False)
    parser.add_argument(
        "--num_clusters",
        help="Number of clusters to be used for clustering approach",
        default=20,
        type=int,
        required=False)
    parser.add_argument(
        "--min_support",
        help="Minimum support to be used for divexplorer approach",
        default=0.03,
        type=float,
        required=False)
    parser.add_argument(
        "--seed",
        help="Seed to be used for reproducibility",
        default=42,
        type=int,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_folder, balancing, dataset='fsc'):
    if balancing:
        df_train = pd.read_csv(os.path.join(df_folder, 'new_data', 'train_data_80.csv'), index_col=None)
        df_valid = pd.read_csv(os.path.join(df_folder, 'new_data', 'valid_data.csv'), index_col=None)
    else:
        df_train = pd.read_csv(os.path.join(df_folder, 'train_data_80.csv'), index_col=None)
        df_valid = pd.read_csv(os.path.join(df_folder, 'valid_data.csv'), index_col=None)
    print("Train size: ", len(df_train))
    print("Valid size: ", len(df_valid))

    if dataset == 'fsc':
        for index in df_train.index:
            df_train.loc[index,'intent'] =  df_train.loc[index,'action'] \
                + df_train.loc[index,'object'] +  df_train.loc[index,'location']

        for index in df_valid.index:
            df_valid.loc[index,'intent'] =  df_valid.loc[index,'action'] \
                + df_valid.loc[index,'object'] +  df_valid.loc[index,'location']

    ## Prepare Labels
    if dataset == 'iemocap':
        labels = df_train['emotion'].unique()
    else:
        labels = df_train['intent'].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    if dataset == 'iemocap':
        for index in range(0,len(df_train)):
            df_train.loc[index,'label'] = label2id[df_train.loc[index,'emotion']]
        for index in range(0,len(df_valid)):
            df_valid.loc[index,'label'] = label2id[df_valid.loc[index,'emotion']]
    else:
        for index in range(0,len(df_train)):
            df_train.loc[index,'label'] = label2id[df_train.loc[index,'intent']]
        for index in range(0,len(df_valid)):
            df_valid.loc[index,'label'] = label2id[df_valid.loc[index,'intent']]
    df_train['label'] = df_train['label'].astype(int)
    df_valid['label'] = df_valid['label'].astype(int)

    return df_train, df_valid, num_labels, label2id, id2label
 

""" Main Program """
if __name__ == '__main__':

    ## Parse command line parameters
    args = parse_cmd_line_params()
    num_epochs = args.epochs
    num_steps = args.steps
    num_wu = args.warmup_steps
    num_gas = args.gradient_accumulation_steps
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)    

    ## Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
            
    ## Train & Test df
    df_train, df_valid, num_labels, label2id, id2label = read_data(
        args.df_folder, 
        args.balancing, 
        args.dataset
        )

    ## Model & Feature Extractor
    model_checkpoint = args.model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    first_run = True

    for epoch in range(num_epochs):

        ## Loading Model
        if not first_run:
            steps = epoch * num_steps
            model_checkpoint = os.path.join(output_dir, f"checkpoint-{steps}")
        print("------------------------------------")
        print(f"Loading model from {model_checkpoint}")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=not first_run
            )
        print("Model loaded successfully!")
        print("------------------------------------\n")

        ## Train & Test Datasets 
        print("----------------------------------")
        print(f"Loading {args.dataset} dataset...")
        max_duration = args.max_duration
        train_dataset = Dataset(
            df_train, 
            feature_extractor, 
            max_duration, 
            balancing=args.balancing, 
            augmentation=args.augmentation
            )
        valid_dataset = Dataset(
            df_valid, 
            feature_extractor, 
            max_duration, 
            balancing=False, 
            augmentation=False
            )
        print("Dataset loaded successfully!")
        print("----------------------------------\n")

        ## Training Arguments
        if first_run:
            training_arguments = define_training_args(output_dir, 
                                                    args.batch, 
                                                    num_steps=num_steps, 
                                                    lr=args.lr, 
                                                    gradient_accumulation_steps=num_gas,
                                                    warmup_steps=num_wu)
        else:
            training_arguments = define_training_args(output_dir, 
                                                    args.batch, 
                                                    num_steps=num_steps*(epoch+1), 
                                                    lr=args.lr, 
                                                    gradient_accumulation_steps=num_gas,
                                                    warmup_steps=num_wu)

        ## Trainer 
        trainer = WeightedTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            balancing=args.balancing)

        ## Train and Evaluate
        print("------------------------------------")
        print(f"Training the model at epoch {epoch+1}...")
        if args.balancing:
            print("Balancing option: activated!")
            if args.approach == 'clustering':
                print("Approach: clustering")
            elif args.approach == 'knn':
                print("Approach: KNN")
            elif args.approach == "random":
                print("Approach: random")
            elif args.approach == 'divexplorer':
                print("Approach: DivExplorer")

        else:
            print("Balancing option: deactivated!")
        print("------------------------------------\n")

        if first_run:
            trainer.train()
            first_run = False
        else:
            trainer.train(resume_from_checkpoint=model_checkpoint)

        predictions = trainer.predict(valid_dataset).predictions
        df_valid['prediction'] = np.argmax(predictions, axis=1) == df_valid['label']
        df_valid['prediction'] = df_valid['prediction'].astype(int)
        df_valid.to_csv(os.path.join(output_dir, f"predictions_{epoch+1}.csv"), index=False)

        if args.balancing:
            print("----------------------------------")
            print("Rebalancing the dataset...")
            weights_rebalancing(
                args.df_folder, 
                df_valid, 
                output_dir,
                args.dataset, 
                args.approach, 
                min_sup=args.min_support,
                num_clusters=args.num_clusters)
            print("----------------------------------\n")

    print("Training completed successfully!")
    print("------------------------------------\n")
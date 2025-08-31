import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.stats
import json
import os
import dateutil, datetime
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS

from package.data_utils.minibatch import PickledMinibatch
from package.data_utils import build_model_Xy_v2
from package.models import EmbeddingLinearHypernetworkModel, EmbeddingSequentialModel
from package.losses import CombinedLoss, NormalizedCosineSimilarity

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import argparse
parser = argparse.ArgumentParser(description='Run Experiments on Dataset')

parser.add_argument(
    '--target_variable',
    type=str,
    choices=["SPEED", "FINISH_TIME"],
    default='SPEED',
    help='The target variable to use.'
)

parser.add_argument(
    '--epochs',
    type=int,
    required=True,
    help='Number of epochs to train the model.'
)

parser.add_argument(
    '--model_type',
    type=str,
    choices=["SEQUENTIAL", "HYPERNETWORK"],
    default='HYPERNETWORK',
    help='Type of model to use.'
)

parser.add_argument(
    '--loss_function',
    type=str,
    choices=["MSE", "MAE", "COSINE_SIMILARITY", "COMBINED_LOSS"],
    default='MAE',
    help='Type of model to use.'
)

parser.add_argument(
    '--data_folder_path',
    type=str,
    required=True,
    help='Path for data folder.'
)

parser.add_argument(
    '--secrets_file',
    type=str,
    required=True,
    help='Path for secret files with wandb api.'
)

parser.add_argument(
    '--dataset',
    type=str,
    choices=["TEST", "VALIDATION"],
    default="VALIDATION",
    help='Dataset to run analysis on'
)

parser.add_argument(
    '--analysis_folder_path',
    type=str,
    required=True,
    help='Folder to save analysis in'
)

parser.add_argument(
    "--heritability_analysis",
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Boolean argument for running heritability analysis'
)

args = parser.parse_args()

wandb_api_key = json.loads(
    open(
        args.secrets_file
    ).read()
)["wandb_api_key"]

wandb_project_name = json.loads(
    open(
        args.secrets_file
    ).read()
)["wandb_project_name"]

wandb.login(key=wandb_api_key)
wandb_api = wandb.Api()

projects = wandb_api.projects()
projects = [project for project in projects if project.name==wandb_project_name]
if(len(projects)==0):
    raise ValueError("No project with macthing name found")
project = projects[0]
project_path = "/".join(project.path)

experiment_filters = {
    "config.target_variable": args.target_variable,
    "config.loss_function": args.loss_function,
    "config.model_type": args.model_type
}

#Find all runs which fit the filter
matching_wandb_runs = wandb_api.runs(path=project_path, filters=experiment_filters)
#Find last reported epoch for the run
matching_wandb_runs_epochs = [x.lastHistoryStep for x in matching_wandb_runs]
#Keep the longest run for further analysis
run = matching_wandb_runs[matching_wandb_runs_epochs.index(max(matching_wandb_runs_epochs))]
#Get run history upto epoch supplied in arguments
run_history = run.history().iloc[:args.epochs]
#Get the minimum validation loss
min_val_loss = run_history["epoch/val_loss"].min()
#Get the epoch (and its timestamp) after which minimum validation loss was recorded
#epoch starting at 1
min_val_loss_epoch = run_history["epoch/val_loss"].argmin()+1

min_val_loss_epoch_timestamp = (
    run_history.iloc[
        run_history["epoch/val_loss"].argmin()
    ]['_timestamp']
)

#Get the list of model save artifacts
model_save_artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type=="model"]

#Find the artifact which created closest to the time of report of min_val_loss_epoch_timestamp
#Essentially finding the artifact which has the model weights that minimize val_loss
artifact_absolute_time_differences = [
    abs(
        dateutil.parser.parse(artifact.created_at).timestamp() - 
        min_val_loss_epoch_timestamp
    ) for artifact in model_save_artifacts
]

best_model_artifact = model_save_artifacts[
    artifact_absolute_time_differences.index(
        min(artifact_absolute_time_differences)
    )
]

#Making sure that our logic of findin artifact using timestamps works
#history_step=epoch-1 hence the assert statement for epoch starting at 0 (min_val_loss_epoch-1)
print(best_model_artifact.history_step, (min_val_loss_epoch-1))
assert(abs(best_model_artifact.history_step-(min_val_loss_epoch-1))<=1)

#Making directory for saving model and analysis results
analysis_save_dir = os.path.join(
    args.analysis_folder_path,
    args.model_type,
    args.loss_function,
    args.target_variable,
    str(args.epochs),
)

print(f"Saving analysis to : {analysis_save_dir}")

os.makedirs(
    analysis_save_dir,
    exist_ok=True
)

#Downloading the model that minimizes loss
best_model_artifact.download(analysis_save_dir)
os.replace(
    os.path.join(
        analysis_save_dir,
        [x.name for x in best_model_artifact.files()][0]
    ),
    os.path.join(
        analysis_save_dir,
        "model.keras"
    )
)

#Load the model weights which minimize validation loss
model = tf.keras.models.load_model(
    os.path.join(
        analysis_save_dir,
        "model.keras"
    )
)

#Saving Model summary
model_summary_filename = os.path.join(
    analysis_save_dir,
    "model_summary.txt"
)
with open(model_summary_filename,'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'+("="*20)+'\n'))
    model.layers[0].summary(print_fn=lambda x: fh.write(x + '\n'+("="*20)+'\n'))
    model.layers[1].summary(print_fn=lambda x: fh.write(x + '\n'+("="*20)+'\n'))


#Save train history - loss and validation loss
train_history = pd.DataFrame()
train_history["epoch"] = run_history['epoch/epoch']+1
train_history["train_loss"] = run_history['epoch/loss']
train_history["validation_loss"] = run_history['epoch/val_loss']
train_history.to_csv(
    os.path.join(
        analysis_save_dir,
        "train_history.csv"
    ),
    index=False
)

#Save plot for train history

#Setting color palette to colorblind for accesebility
sns.set_palette(sns.color_palette("colorblind"))

fig,ax = plt.subplots(1,1)

#Plotting Validation Loss
sns.lineplot(
    x=train_history["epoch"].values[1:]+1, 
    y=train_history['validation_loss'].values[1:],
    label="Validation Loss",
    ax=ax
)


#Plotting Train Loss
sns.lineplot(
    x=train_history["epoch"].values[1:]+1, 
    y=train_history['train_loss'].values[1:],
    label="Train Loss",
    ax=ax
)

#Labels, limits, ticks, etc. Code for pretty graphs.
ax.set_xlabel("Epochs")

#Some latex string mappings required for making graphs
plotting_loss_latex_str_map = {
    "MSE":"$L_{M.S.E.}$",
    "MAE":"$L_{M.A.E.}$",
    "COSINE_SIMILARITY":"$L_{C.S.}$",
    "COMBINED_LOSS":"$L_{C}(\\alpha_1=0,\\alpha_2=1)$"
}

ax.set_ylabel("Loss Function "+ plotting_loss_latex_str_map[args.loss_function])
ax.set_xticks(list(range(0, args.epochs+1,5)))
ax.set_xlim(0,args.epochs+1)
fig.tight_layout()
fig.set_figheight(4) #HARDCODED
fig.set_figwidth(10) #HARDCODED

fig.savefig(
    os.path.join(
        analysis_save_dir,
        "train_history.png"
    )
)

#Generating predictions using the best model weights

#Reading test/val dataset
if(args.dataset=="VALIDATION"):
    dataset = pd.read_parquet(
        os.path.join(
            args.data_folder_path,
            f"VALIDATION_{args.target_variable}.pq"
        )
    )

if(args.dataset=="TEST"):
    dataset = pd.read_parquet(
        os.path.join(
            args.data_folder_path,
            f"TEST_{args.target_variable}.pq"
        )
    )

#Reading model config
runner_categorical_variables = run.config['horse_embedding_model_kwargs']['categorical_feature_names']
race_categorical_variables = run.config['race_embedding_model_kwargs']['categorical_feature_names']

#Generating covariates
X,y = build_model_Xy_v2(
    dataset,
    runner_categorical_variables,
    race_categorical_variables
)

#Generating predictions using the model
y_hat = model.predict(X, batch_size=1024)

#Saving predictions with observed, race_id, runner_id
scoring_dataset = dataset[['race_id', 'runner_id', 'y']].copy()
scoring_dataset['y_hat'] = y_hat.flatten()
scoring_dataset.to_csv(
    os.path.join(
        analysis_save_dir,
        f"{args.dataset}_scoring_dataset.csv"
    ),
    index=False
)

#Calculating Race-Wise Spearman and Pearson
#Selecting races with atleast 3 runners
more_than_2_subset = (
    scoring_dataset[
        ['race_id', 'y', 'y_hat']
    ]
).groupby("race_id").filter(lambda x : x.shape[0]>2)

#DataFrame to save race-wise spearman, pearson, number of runners
corr_spr_with_count = pd.DataFrame()
corr_spr_with_count['pearson'] = more_than_2_subset.groupby("race_id").apply(lambda x : np.corrcoef(x["y"].values, x["y_hat"].values)[1,0])
corr_spr_with_count["spearman"] = more_than_2_subset.groupby("race_id").apply(lambda x : scipy.stats.spearmanr(x["y"].values, x["y_hat"].values).statistic)
corr_spr_with_count['runner_count'] = more_than_2_subset.groupby("race_id").count()['y']

#Dropping rows with NAs caused by 0 variance in either observed or predicted values
#Can happen from time to time but is very rare.
corr_spr_with_count = corr_spr_with_count.dropna()

#Saving race-wise pearson, spearman, runner_count
corr_spr_with_count.to_csv(
    os.path.join(
        analysis_save_dir,
        f"{args.dataset}_race_wise_pearson_spearman_count.csv"
    ),
    index=False
)

#Saving stats of race-wise pearson, spearman, runner_count
corr_spr_with_count.describe().to_csv(
    os.path.join(
        analysis_save_dir,
        f"{args.dataset}_race_wise_pearson_spearman_count_statistical_summary.csv"
    )
)

def corr_meta_analysis(corr_df, corr_column, significance=.95):
    norm_ppf = scipy.stats.norm().ppf(1-(1-significance)/2)
    corr_df = corr_df.dropna()
    n = corr_df['runner_count'].values
    corr = corr_df[corr_column].values
    z = 0.5*(np.log(1+corr+1e-5)-np.log(1-corr+1e-5))
    zp = (z*(n-3)).sum()/((n-3).sum())
    se = np.sqrt(1/((n-3).sum()))
    return zp.round(5), ((zp-(norm_ppf*se)).round(5), (zp+(norm_ppf*se)).round(5))

pearson_meta_analysis = corr_meta_analysis(corr_spr_with_count, "pearson")
spearman_meta_analysis = corr_meta_analysis(corr_spr_with_count, "spearman")

pd.DataFrame(
    data = np.array(
        [
            ["2.5 % CI", pearson_meta_analysis[1][0], spearman_meta_analysis[1][0]],
            ["Estimate", pearson_meta_analysis[0], spearman_meta_analysis[0]],
            ["97.5 % CI", pearson_meta_analysis[1][1], spearman_meta_analysis[1][1]],
        ]
    ),
    columns = ["Statistic", "Pearson zeta_O", "Spearman zeta_O"]
).to_csv(os.path.join(
        analysis_save_dir,
        f"{args.dataset}_corr_meta_analysis.csv"
    ),
    index=False
)


if(not args.heritability_analysis):
    exit()


#Heritability Analysis begins here
horse_embeddings_normalized = model.layers[1].get_embeddings("runner_id")
horse_embeddings_normalized = horse_embeddings_normalized/np.linalg.norm(horse_embeddings_normalized, axis=1, keepdims=True)


gean_data = pd.read_parquet(
    os.path.join(
        args.data_folder_path,
        f"GEAN_DATA.pq"
    )
).drop_duplicates("runner_id")

npr = np.random.RandomState(0)
gean_data['random_sire_id'] = npr.choice(
    gean_data['sire_id'].unique(),
    size = gean_data.shape[0],
)
gean_data['random_dam_id'] = npr.choice(
    gean_data['dam_id'].unique(),
    size = gean_data.shape[0],
)

cs_sample_df = pd.DataFrame()

cs_sample_df["offspring_dam_permuted"] = (
    horse_embeddings_normalized[gean_data['runner_id'].values] *
    horse_embeddings_normalized[gean_data['random_dam_id'].values]
).sum(axis=1)

cs_sample_df["offspring_dam_unpermuted"] = (
    horse_embeddings_normalized[gean_data['runner_id'].values] *
    horse_embeddings_normalized[gean_data['dam_id'].values]
).sum(axis=1)

cs_sample_df["offspring_sire_permuted"] = (
    horse_embeddings_normalized[gean_data['runner_id'].values] *
    horse_embeddings_normalized[gean_data['random_sire_id'].values]
).sum(axis=1)

cs_sample_df["offspring_sire_unpermuted"] = (
    horse_embeddings_normalized[gean_data['runner_id'].values] *
    horse_embeddings_normalized[gean_data['sire_id'].values]
).sum(axis=1)

cs_sample_df.describe().to_csv(
    os.path.join(
        analysis_save_dir,
        "cosine_sim_heritability_analysis.csv"
    )
)

def plot_cosine_similarity(parent, df):
    fig, ax = plt.subplots(1,2, sharey=True)
    unperm = cs_sample_df[f"offspring_{parent}_unpermuted"]
    perm = cs_sample_df[f"offspring_{parent}_permuted"]
    unperm.plot.hist(
        bins=100,
        xlim=(-1,1),
        xlabel="Cosine Similarity",
        density=True,
        ax=ax[1],
        title="Unpermuted Cosine Similarity"
    )
    
    perm.plot.hist(
        bins=100,
        xlim=(-1,1),
        xlabel="Cosine Similarity",
        density=True,
        ax=ax[0],
        title="Permuted Cosine Similarity"
    )

    fig.tight_layout()
    fig.set_figheight(4) #HARDCODED
    fig.set_figwidth(10) #HARDCODED
    return fig,ax

fig, ax = plot_cosine_similarity("sire", cs_sample_df)

fig.savefig(
    os.path.join(
        analysis_save_dir,
        "sire_cs_hist.png"
    )
)

fig, ax = plot_cosine_similarity("dam", cs_sample_df)

fig.savefig(
    os.path.join(
        analysis_save_dir,
        "dam_cs_hist.png"
    )
)

ks_test_sire = scipy.stats.kstest(
    cs_sample_df["offspring_sire_permuted"].values,
    cs_sample_df["offspring_sire_unpermuted"].values
)

ks_test_dam = scipy.stats.kstest(
    cs_sample_df["offspring_dam_permuted"].values,
    cs_sample_df["offspring_dam_unpermuted"].values
)

ks_test_file = open(
    os.path.join(
        analysis_save_dir,
        "ks_test_results.txt"
    ),
    "w"
)

ks_test_file.write("SIRE KS TEST RESULTS\n\n")
ks_test_file.write(str(ks_test_sire))

ks_test_file.write("\n\n"+("="*50)+"\n\n")

ks_test_file.write("DAM KS TEST RESULTS\n\n")
ks_test_file.write(str(ks_test_dam))

ks_test_file.close()

horse_embeddings = model.layers[1].get_embeddings("runner_id", normalize=False)

parent_phenotype = np.concatenate(
    [
        horse_embeddings[gean_data["sire_id"].values],
        horse_embeddings[gean_data["dam_id"].values],
        np.ones(shape=(gean_data.shape[0],1)),
    ],
    axis=1
)

child_phenotype = horse_embeddings[gean_data["runner_id"].values]
noise = np.random.normal(size=horse_embeddings.shape)

lm_summary_file_text = open(
    os.path.join(
        analysis_save_dir,
        "phenotype_linear_model_summary.txt"
    ),
    "w"
)

lm_summary_file_latex = open(
    os.path.join(
        analysis_save_dir,
        "phenotype_linear_model_summary.tex"
    ),
    "w"
)




for dim_counter in range(0,child_phenotype.shape[1]):
    ols_fit = OLS(
        child_phenotype[:, dim_counter],
        parent_phenotype            
    ).fit()

    lm_summary_file_text.write(ols_fit.summary().as_text())
    lm_summary_file_text.write("\n\n"+("="*100)+"\n\n")
    
    lm_summary_file_latex.write(ols_fit.summary().as_latex())
    lm_summary_file_latex.write("\n\n\n\n")

    resid_std = np.sqrt(np.power(ols_fit.resid, 2).mean())
    print(resid_std)
    noise[:,dim_counter] = noise[:,dim_counter]*resid_std

lm_summary_file_text.close()
lm_summary_file_latex.close()

horse_embeddings_hat = horse_embeddings+noise
model.layers[1].layers[0].set_weights([horse_embeddings_hat])
print((model.layers[1].get_embeddings("runner_id", normalize=False)-horse_embeddings).std(axis=0))

#Generating predictions using the model
ep_y_hat = model.predict(X, batch_size=1024)

#Saving predictions with observed, race_id, runner_id
ep_scoring_dataset = dataset[['race_id', 'runner_id', 'y']].copy()
ep_scoring_dataset['y_hat'] = ep_y_hat.flatten()


#Calculating Race-Wise Spearman and Pearson
#Selecting races with atleast 3 runners
ep_more_than_2_subset = (
    ep_scoring_dataset[
        ['race_id', 'y', 'y_hat']
    ]
).groupby("race_id").filter(lambda x : x.shape[0]>2)

#DataFrame to save race-wise spearman, pearson, number of runners
ep_corr_spr_with_count = pd.DataFrame()
ep_corr_spr_with_count['pearson'] = ep_more_than_2_subset.groupby("race_id").apply(lambda x : np.corrcoef(x["y"].values, x["y_hat"].values)[1,0])
ep_corr_spr_with_count["spearman"] = ep_more_than_2_subset.groupby("race_id").apply(lambda x : scipy.stats.spearmanr(x["y"].values, x["y_hat"].values).statistic)
ep_corr_spr_with_count['runner_count'] = ep_more_than_2_subset.groupby("race_id").count()['y']

#Dropping rows with NAs caused by 0 variance in either observed or predicted values
#Can happen from time to time but is very rare.
ep_corr_spr_with_count = ep_corr_spr_with_count.dropna()

#Saving race-wise pearson, spearman, runner_count for error prop
ep_corr_spr_with_count.to_csv(
    os.path.join(
        analysis_save_dir,
        f"ep_{args.dataset}_race_wise_pearson_spearman_count.csv"
    ),
    index=False
)

#Saving stats of race-wise pearson, spearman, runner_count for error prop
ep_corr_spr_with_count.describe().to_csv(
    os.path.join(
        analysis_save_dir,
        f"ep_{args.dataset}_race_wise_pearson_spearman_count_statistical_summary.csv"
    )
)



ep_pearson_meta_analysis = corr_meta_analysis(ep_corr_spr_with_count, "pearson")
ep_spearman_meta_analysis = corr_meta_analysis(ep_corr_spr_with_count, "spearman")

pd.DataFrame(
    data = np.array(
        [
            ["2.5 % CI", ep_pearson_meta_analysis[1][0], ep_spearman_meta_analysis[1][0]],
            ["Estimate", ep_pearson_meta_analysis[0], ep_spearman_meta_analysis[0]],
            ["97.5 % CI", ep_pearson_meta_analysis[1][1], ep_spearman_meta_analysis[1][1]],
        ]
    ),
    columns = ["Statistic", "Pearson zeta_O", "Spearman zeta_O"]
).to_csv(os.path.join(
        analysis_save_dir,
        f"ep_{args.dataset}_corr_meta_analysis.csv"
    ),
    index=False
)

exit()



#model_type_latex_map = {"SEQUENTIAL": "Sequential", "HYPERNETWORK":"Hypernetwork"}
#target_variable_latex_map = {"FINISH_TIME": "$ln($Finish Time$)$", "SPEED": "$ln($Average Speed$)$"}
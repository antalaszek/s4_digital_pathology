import torch
import utils
import models
import random
import argparse
from s4dataset import S4Dataset
from torch.utils.data import DataLoader
from custom_utils import WSIDataModule, RandomStratifiedGroupKFoldTrainValTest

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="Path to the .yaml config file."
)
parser.add_argument(
    "--fold",
    required=False,
    type=int,
    default=None,
    help="Fold on which to launch training.",
)
args = parser.parse_args()

# read the config file and update if needed
config = utils.read_config(args.config)
utils.check_config(config)
if args.fold is not None:
    config.data.fold = args.fold
random.seed(config.seed)
torch.manual_seed(config.seed)

# create the model (note the S4Model is not yet compatible with MPS)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on {str(device)}.")
model = getattr(models, config.model.model_type)(config).to(device)

# create the datasets and dataloaders
wsi_dm = WSIDataModule(
    "/home/amoszyns/workdir/clam_new/features_comb_aug/pt_files/",
    augmentations=[],
    fold_id=config.data.fold,
    split_strategy=RandomStratifiedGroupKFoldTrainValTest(10, 9, random_state=123),
    weighted_random_sampler=True,
    data_subset_regex="linz",
    num_workers=30,
    save_path=f"dataset/{config.data.fold}",
)
wsi_dm.setup()
train_dataloader = wsi_dm.train_dataloader()
# DataLoader(
#     train_dataset, batch_size=config.data.batch_size, shuffle=True
# )
val_dataloader = wsi_dm.val_dataloader()
# DataLoader(
#     val_dataset, batch_size=config.data.batch_size, shuffle=False
# )

utils.train(config, model, device, train_dataloader, val_dataloader)

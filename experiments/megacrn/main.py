import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.models.megacrn import MegaCRN
from src.engines.megacrn_engine import MegaCRN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--rnn_size", type=int, default=64)
    parser.add_argument("--hyperGNN_dim", type=int, default=32)
    parser.add_argument("--node_dim", type=int, default=24)
    parser.add_argument("--tanhalpha", type=int, default=3)
    parser.add_argument("--cl_decay_step", type=int, default=2000)
    parser.add_argument("--step_size", type=int, default=2500)
    parser.add_argument("--tpd", type=int, default=96)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    args.model_name = __file__.split("/")[-2]
    folder_name = "{}_{}".format(args.dataset, args.years)
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info("Adj path: " + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, "doubletransition")

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = MegaCRN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        mem_dim=args.node_dim,
        rnn_units=64
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    

    engine = MegaCRN_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        step_size=args.step_size,
        horizon=args.horizon,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate("test")
        engine.evaluate("shift")


if __name__ == "__main__":
    main()

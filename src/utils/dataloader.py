import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp


class DataLoader(object):
    def __init__(
        self, data, idx, seq_len, horizon, bs, logger, node_embed, pad_last_sample=False
    ):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.node_embed = node_embed
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info(
            "Sample num: "
            + str(self.idx.shape[0])
            + ", Batch num: "
            + str(self.num_batch)
        )

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon

    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind:end_ind, ...]

                x_shape = (
                    len(idx_ind),
                    self.seq_len,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                x_shared = mp.RawArray("f", int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype="f").reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray("f", int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype="f").reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = (
                        start_index + chunk_size if i < num_threads - 1 else array_size
                    )
                    thread = threading.Thread(
                        target=self.write_to_shared_array,
                        args=(x, y, idx_ind, start_index, end_index),
                    )
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, "his.npz"))
    logger.info("Data shape: " + str(ptr["data"].shape))
    if args.shift == False:
        node_embed = np.load(os.path.join("data", args.dataset, f"{args.dataset}_emb1.npy"))
    else:
        node_embed = np.load(os.path.join("data", args.dataset, f"{args.dataset}_emb2.npy"))
    dataloader = {}
    for cat in ["train", "val", "test", "shift"]:
        idx = np.load(os.path.join(data_path, "idx_" + cat + ".npy"))
        dataloader[cat + "_loader"] = DataLoader(
            ptr["data"][..., : args.input_dim],
            idx,
            args.seq_len,
            args.horizon,
            args.bs,
            logger,
            node_embed,
        )

    scaler = StandardScaler(mean=ptr["mean"], std=ptr["std"])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)



def get_dataset_info(dataset):
    base_dir =  "/data1/chenjiyuan/adaptive_embed/data/"
    other_dir = "/data1/cjy/st_mamba/data/"
    d = {
        "CA": [other_dir + "ca", other_dir + "ca/ca_rn_adj.npy", 8600],
        "GLA": [other_dir + "gla", other_dir + "gla/gla_rn_adj.npy", 3834],
        "GBA": [other_dir + "gba", other_dir + "gba/gba_rn_adj.npy", 2352],
        "SD": [other_dir + "sd", other_dir + "sd/sd_rn_adj.npy", 716],
        "PEMS04": [base_dir + "PEMS04/2018", base_dir + "PEMS04/adj.npy", 307],
        "PEMS07": [base_dir + "PEMS07/2017", base_dir + "PEMS07/adj.npy", 883],
        "PEMS03": [base_dir + "PEMS03/2018", base_dir + "PEMS03/adj.npy", 358],
        "PEMS08": [base_dir + "PEMS08/2016", base_dir + "PEMS08/adj.npy", 170],
        "Bike_Chicago": [
            base_dir + "Bike_Chicago",
            base_dir + "Bike_Chicago/adj.npy",
            585,
        ],
        "Bus_NYC": [base_dir + "Bus_NYC", base_dir + "Bus_NYC/adj.npy", 226],
        "Taxi_Chicago": [
            base_dir + "Taxi_Chicago",
            base_dir + "Taxi_Chicago/adj.npy",
            77,
        ],
        "Taxi_NYC": [base_dir + "Taxi_NYC", base_dir + "Taxi_NYC/adj.npy", 263],
        "Bike_DC": [base_dir + "Bike_DC", base_dir + "Bike_DC/adj.npy", 532],
        "Bike_NYC": [base_dir + "Bike_NYC", base_dir + "Bike_NYC/adj.npy", 532],
    }
    assert dataset in d.keys()
    return d[dataset]

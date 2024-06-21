import time
import torch
import numpy as np

import dgl
from torch.utils.data import DataLoader
from collections.abc import Mapping

from dgl.dataloading import create_tensorized_dataset, WorkerInitWrapper, remove_parent_storage_columns
from dgl.utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads, get_num_threads,
    get_numa_nodes_cores, context_of, dtype_of)

from dgl.dataloading.dataloader import _TensorizedDatasetIter, _divide_by_worker
from dgl import DGLHeteroGraph
from dgl.frame import LazyFeature
from dgl.storages import wrap_storage
from dgl.dataloading.base import BlockSampler, as_edge_prediction_sampler
from dgl import backend as F
from dgl.distributed import DistGraph
from dgl.multiprocessing import call_once_and_share
import torch.distributed as dist


from collections import Counter


def _get_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device

class _SimTensorizedDatasetIter(object):
    def __init__(self, dataset, batch_size, drop_last, mapping_keys, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mapping_keys = mapping_keys
        self.index = 0
        self.shuffle = shuffle

    # For PyTorch Lightning compatibility
    def __iter__(self):
        return self

    # Need to modify this function
    def _next_indices(self):
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.index + self.batch_size
        if end_idx > num_items:
            if self.drop_last:
                raise StopIteration
            end_idx = num_items
        batch = self.dataset[self.index : end_idx]
        self.index += self.batch_size

        return batch

    def __next__(self):
        batch = self._next_indices()
        print("Tensorized batch: ", batch)
        if self.mapping_keys is None:
            # clone() fixes #3755, probably.  Not sure why.  Need to take a look afterwards.
            return batch.clone()

        # convert the type-ID pairs to dictionary
        type_ids = batch[:, 0]
        indices = batch[:, 1]
        _, type_ids_sortidx = torch.sort(type_ids, stable=True)
        type_ids = type_ids[type_ids_sortidx]
        indices = indices[type_ids_sortidx]
        type_id_uniq, type_id_count = torch.unique_consecutive(
            type_ids, return_counts=True
        )
        type_id_uniq = type_id_uniq.tolist()
        type_id_offset = type_id_count.cumsum(0).tolist()
        type_id_offset.insert(0, 0)
        id_dict = {
            self.mapping_keys[type_id_uniq[i]]: indices[
                type_id_offset[i] : type_id_offset[i + 1]
            ].clone()
            for i in range(len(type_id_uniq))
        }
        return id_dict




def create_sim_tensorized_dataset(
    indices,
    batch_size,
    drop_last,
    use_ddp,
    ddp_seed,
    shuffle,
    use_shared_memory,
):
    """Converts a given indices tensor to a TensorizedDataset, an IterableDataset
    that returns views of the original tensor, to reduce overhead from having
    a list of scalar tensors in default PyTorch DataLoader implementation.
    """
    if use_ddp:
        # DDP always uses shared memory
        return DDPTensorizedDataset(
            indices, batch_size, drop_last, ddp_seed, shuffle
        )
    else:
        return SimTensorizedDataset(
            indices, batch_size, drop_last, shuffle, use_shared_memory
        )



class SimTensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.
    """

    def __init__(
        self, indices, batch_size, drop_last, shuffle, use_shared_memory
    ):
        if isinstance(indices, Mapping):
            self._mapping_keys = list(indices.keys())
            self._device = next(iter(indices.values())).device
            self._id_tensor = _get_id_tensor_from_mapping(
                indices, self._device, self._mapping_keys
            )
        else:
            self._id_tensor = indices
            self._device = indices.device
            self._mapping_keys = None
        # Use a shared memory array to permute indices for shuffling.  This is to make sure that
        # the worker processes can see it when persistent_workers=True, where self._indices
        # would not be duplicated every epoch.
        self._indices = torch.arange(
            self._id_tensor.shape[0], dtype=torch.int64
        )
        if use_shared_memory:
            self._indices.share_memory_()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._shuffle = shuffle

    def shuffle(self):
        """Shuffle the dataset."""
        np.random.shuffle(self._indices.numpy())

    def __iter__(self):
        indices = _divide_by_worker(
            self._indices, self.batch_size, self.drop_last
        )
        id_tensor = self._id_tensor[indices]
        return _SimTensorizedDatasetIter(
            id_tensor,
            self.batch_size,
            self.drop_last,
            self._mapping_keys,
            self._shuffle,
        )

    def __len__(self):
        num_samples = self._id_tensor.shape[0]
        return (
            num_samples + (0 if self.drop_last else (self.batch_size - 1))
        ) // self.batch_size



class CollateWrapper(object):
    def __init__(self, sample_func, g,  device):
        self.sample_func = sample_func
        self.g = g
        self.device = device
        self.pin_memory = True


    def __call__(self, items):
        graph_device = getattr(self.g, 'device', None)   
        items = recursive_apply(items, lambda x: x.to(self.device))
        print("items: ", items)
        batch = self.sample_func(self.g, items)
        print("batch: ", batch)
        return batch


class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler

    def __iter__(self):
        return self

    def __next__(self):
        print("next")
        cur_it = self.dataloader_it
        return next(cur_it)


class Simulation_Loader(torch.utils.data.DataLoader):
    def __init__(self, graph, indices, graph_sampler, batch_size, dim, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None,

                 **kwargs):


        use_uva = False
#        self.GIDS_Loader = GIDS
        self.dim = dim

        if isinstance(kwargs.get('collate_fn', None), CollateWrapper):
            assert batch_size is None       # must be None
            # restore attributes
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_alternate_streams = use_alternate_streams
            self.use_uva = use_uva
            kwargs['batch_size'] = None
            super().__init__(**kwargs)
            return


        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')
  
        self.graph = graph
        self.indices = indices     
        num_workers = kwargs.get('num_workers', 0)


        indices_device = None
        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                indices_device = next(iter(indices.values())).device
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        if indices_device is None:
            if not hasattr(indices, 'device'):
                raise AttributeError('Custom indices dataset requires a \"device\" \
                attribute indicating where the indices is.')
            indices_device = indices.device

        if device is None:     
            device = torch.cuda.current_device()
        self.device = _get_device(device)
        
        #pin graph
        if not self.graph._graph.is_pinned():
            print("\t\tPinning graph")
            self.graph._graph.pin_memory_()
            
        else:
            print('\t\tNot pinning Graph')


        # Sanity check - we only check for DGLGraphs.
        if isinstance(self.graph, DGLHeteroGraph):            
            self.graph.create_formats_()
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()
            

            # Check use_alternate_streams
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == 'cuda' and self.graph.device.type == 'cpu' and
                    not use_uva)

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            print("create tensorized dataset")
            self.dataset = create_sim_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle,
                kwargs.get('persistent_workers', False))
        else:
            print("indicies")
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams


        self.cpu_affinity_enabled = False

        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        self.other_storages = {}
        print("self dataset: ", self.dataset)

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__())

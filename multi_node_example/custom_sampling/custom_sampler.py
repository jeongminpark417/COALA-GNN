import dgl
from dgl import backend as F, utils
from dgl.base import EID, NID
from dgl.heterograph import DGLGraph, DGLBlock
from dgl.transforms import to_block
from dgl.utils import get_num_threads
from dgl.dataloading import BlockSampler

from dgl._ffi.capi import *


from custom_neighbor import sample_neighbors2

import torch

from collections import defaultdict
from collections.abc import Mapping


class Hot_Node_BFS(BlockSampler):
    def __inint__(
        self,
        edge_dir="in",
    ):
        self.g = None


def to_block_with_meta(g, dst_nodes=None, dst_nodes_meta=None, include_dst_in_src=True, src_nodes=None):
    if dst_nodes is None:
        # Find all nodes that appeared as destinations
        dst_nodes = defaultdict(list)
        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            dst_nodes[etype[2]].append(dst)
        dst_nodes = {
            ntype: F.unique(F.cat(values, 0))
            for ntype, values in dst_nodes.items()
        }
    elif not isinstance(dst_nodes, Mapping):
        # dst_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise DGLError(
                "Graph has more than one node type; please specify a dict for dst_nodes."
            )
        dst_nodes = {g.ntypes[0]: dst_nodes}
 
        

    dst_node_ids = [
        utils.toindex(dst_nodes.get(ntype, []), g._idtype_str).tousertensor(
            ctx=F.to_backend_ctx(g._graph.ctx)
        )
        for ntype in g.ntypes
    ]
    dst_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in dst_node_ids]

    for d in dst_node_ids_nd:
        if g._graph.ctx != d.ctx:
            raise ValueError("g and dst_nodes need to have the same context.")

    src_node_ids = None
    src_node_ids_nd = None
    if src_nodes is not None and not isinstance(src_nodes, Mapping):
        # src_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise DGLError(
                "Graph has more than one node type; please specify a dict for src_nodes."
            )
        src_nodes = {g.ntypes[0]: src_nodes}
        src_node_ids = [
            F.copy_to(
                F.tensor(src_nodes.get(ntype, []), dtype=g.idtype),
                F.to_backend_ctx(g._graph.ctx),
            )
            for ntype in g.ntypes
        ]
        src_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in src_node_ids]

        for d in src_node_ids_nd:
            if g._graph.ctx != d.ctx:
                raise ValueError(
                    "g and src_nodes need to have the same context."
                )
    else:

        # use an empty list to signal we need to generate it
        src_node_ids_nd = []

    new_graph_index, src_nodes_ids_nd, induced_edges_nd = _CAPI_DGLToBlock(
        g._graph, dst_node_ids_nd, include_dst_in_src, src_node_ids_nd
    )

    # The new graph duplicates the original node types to SRC and DST sets.
    new_ntypes = (g.ntypes, g.ntypes)
    new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
    assert new_graph.is_unibipartite  # sanity check

    src_node_ids = [F.from_dgl_nd(src) for src in src_nodes_ids_nd]
    edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges_nd]

    node_frames = utils.extract_node_subframes_for_block(
        g, src_node_ids, dst_node_ids
    )
    edge_frames = utils.extract_edge_subframes(g, edge_ids)
    utils.set_new_frames(
        new_graph, node_frames=node_frames, edge_frames=edge_frames
    )




    # TEST
    # print("src nodes ids nd: ", src_node_ids)
    # print("new_graph_index: ", new_graph_index)

    # print("induced edge nd: ", edge_ids)
    # print("edge frame: ", edge_frames)


    return new_graph

    # # Initialize src_nodes_meta based on the mapping
    # if dst_nodes_meta is not None:
    #     # Initialize src_nodes_meta with zeros or any default value
    #     src_nodes_meta = torch.zeros_like(dst_nodes_meta)

    #     # Assuming the dst_nodes_meta is given for dst_nodes and you have the mapping now
    #     src_nodes = [F.from_dgl_nd(src) for src in src_nodes_ids_nd][0]  # Assuming single node type
    #     dst_nodes_tensor = dst_node_ids[0]  # Assuming single node type
    #     for src_id, dst_id in enumerate(dst_nodes_tensor.tolist()):
    #         #print("src id: ", src_id, " dst_id: ", dst_id)
    #         src_nodes_meta[src_id] = dst_nodes_meta[dst_id]

    # The new graph duplicates the original node types to SRC and DST sets.
  


class NeighborSampler2(BlockSampler):
   

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=False,
        hot_nodes = None
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.hot_nodes = hot_nodes

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        
        init_hot_nodes = False
        prev_hot_nodes = None

        for fanout in reversed(self.fanouts):
            frontier = sample_neighbors2(
                g,
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            # print("frontiers: ", frontier)
            # print("seed nodes: ", seed_nodes)
            #block = to_block(frontier, seed_nodes)
            block = to_block_with_meta(frontier, seed_nodes)
            # print("block: ", block)


            if(init_hot_nodes ==  False):
                assert (self.hot_nodes != None)
                cpu_seed = seed_nodes.to("cpu")
                prev_hot_nodes = self.hot_nodes[cpu_seed].to(seed_nodes.device)
                #print("prev_hot_nodes: ", prev_hot_nodes)


            #temp = []
            #print("num nodes: ", g.number_of_nodes())
            # print("seed nodes: ", seed_nodes)
            # for i in range((g.number_of_nodes())):
            #     temp.append(i)
            # meta = torch.Tensor(temp)
            # #print("meta: ", meta)
            # block = to_block_with_meta(frontier, seed_nodes,meta)
            

     

            # for src, dst in zip(src_ids.tolist(), dst_ids.tolist()):
            #     print(f"Block Source: {src}, Destination: {dst}")


            #print("block: ", block)

            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]

            cur_hot_nodes = torch.full_like(seed_nodes, fill_value=-1, dtype=torch.int64, device=seed_nodes.device)

            src_ids, dst_ids = block.edges()
            # print("Block src ids: ", src_ids)
            # print("Block dst ids: ", dst_ids)



            cur_hot_nodes[src_ids] = prev_hot_nodes[dst_ids]

           

            block.hot_nodes = cur_hot_nodes
            if(init_hot_nodes ==  False):
                block.src_hot_nodes = prev_hot_nodes

            prev_hot_nodes = cur_hot_nodes

            #print("prev: ", prev_hot_nodes)

            blocks.insert(0, block)
            init_hot_nodes = True

       # print("seed nodes: ", seed_nodes)
        #blocks[-1].hot_nodes = prev_hot_nodes
        return seed_nodes, output_nodes, blocks






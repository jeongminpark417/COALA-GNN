import ctypes

import dgl

from dgl import utils

from dgl import backend as F, ndarray as nd, utils
#from .. import backend as F, ndarray as nd, utils

from dgl._ffi.function import _init_api, get_global_func
#from .._ffi.function import _init_api

from dgl.base import DGLError, EID
from dgl.heterograph import DGLBlock, DGLGraph
#from .utils import EidExcluder

#from dgl import sampling

#from libdgl import _CAPI_DGLSampleNeighbors

def _prepare_edge_arrays(g, arg):
    """Converts the argument into a list of NDArrays.

    If the argument is already a list of array-like objects, directly do the
    conversion.

    If the argument is a string, converts g.edata[arg] into a list of NDArrays
    ordered by the edge types.
    """
    if isinstance(arg, list) and len(arg) > 0:
        if isinstance(arg[0], nd.NDArray):
            return arg
        else:
            # The list can have None as placeholders for empty arrays with
            # undetermined data type.
            dtype = None
            ctx = None
            result = []
            for entry in arg:
                if F.is_tensor(entry):
                    result.append(F.to_dgl_nd(entry))
                    dtype = F.dtype(entry)
                    ctx = F.context(entry)
                else:
                    result.append(None)

            result = [
                F.to_dgl_nd(F.copy_to(F.tensor([], dtype=dtype), ctx))
                if x is None
                else x
                for x in result
            ]
            return result
    elif arg is None:
        return [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        arrays = []
        for etype in g.canonical_etypes:
            if arg in g.edges[etype].data:
                arrays.append(F.to_dgl_nd(g.edges[etype].data[arg]))
            else:
                arrays.append(nd.array([], ctx=nd.cpu()))
        return arrays


def _sample_neighbors2(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    _dist_training=False,
    exclude_edges=None,
    fused=False,
    mapping=None,
):
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {g.ntypes[0]: nodes}

    nodes = utils.prepare_tensor_dict(g, nodes, "nodes")
    if len(nodes) == 0:
        raise ValueError(
            "Got an empty dictionary in the nodes argument. "
            "Please pass in a dictionary with empty tensors as values instead."
        )
    device = utils.context_of(nodes)
    ctx = utils.to_dgl_context(device)
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.array([], ctx=ctx))

    if isinstance(fanout, nd.NDArray):
        fanout_array = fanout
    else:
        if not isinstance(fanout, dict):
            fanout_array = [int(fanout)] * len(g.etypes)
        else:
            if len(fanout) != len(g.etypes):
                raise DGLError(
                    "Fan-out must be specified for each edge type "
                    "if a dict is provided."
                )
            fanout_array = [None] * len(g.etypes)
            for etype, value in fanout.items():
                fanout_array[g.get_etype_id(etype)] = value
        fanout_array = F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64))

    prob_arrays = _prepare_edge_arrays(g, prob)

    excluded_edges_all_t = []
    if exclude_edges is not None:
        if not isinstance(exclude_edges, dict):
            if len(g.etypes) > 1:
                raise DGLError(
                    "Must specify etype when the graph is not homogeneous."
                )
            exclude_edges = {g.canonical_etypes[0]: exclude_edges}
        exclude_edges = utils.prepare_tensor_dict(g, exclude_edges, "edges")
        for etype in g.canonical_etypes:
            if etype in exclude_edges:
                excluded_edges_all_t.append(F.to_dgl_nd(exclude_edges[etype]))
            else:
                excluded_edges_all_t.append(nd.array([], ctx=ctx))

    if fused:
        if _dist_training:
            raise DGLError(
                "distributed training not supported in fused sampling"
            )
        cpu = F.device_type(g.device) == "cpu"
        if isinstance(nodes, dict):
            for ntype in list(nodes.keys()):
                if not cpu:
                    break
                cpu = cpu and F.device_type(nodes[ntype].device) == "cpu"
        else:
            cpu = cpu and F.device_type(nodes.device) == "cpu"
        if not cpu or F.backend_name != "pytorch":
            raise DGLError(
                "Only PyTorch backend and cpu is supported in fused sampling"
            )

        if mapping is None:
            mapping = {}
        mapping_name = "__mapping" + str(os.getpid())
        if mapping_name not in mapping.keys():
            mapping[mapping_name] = [
                torch.LongTensor(g.num_nodes(ntype)).fill_(-1)
                for ntype in g.ntypes
            ]

        subgidx, induced_nodes, induced_edges = _CAPI_DGLSampleNeighborsFused(
            g._graph,
            nodes_all_types,
            [F.to_dgl_nd(m) for m in mapping[mapping_name]],
            fanout_array,
            edge_dir,
            prob_arrays,
            excluded_edges_all_t,
            replace,
        )
        for mapping_vector, src_nodes in zip(
            mapping[mapping_name], induced_nodes
        ):
            mapping_vector[F.from_dgl_nd(src_nodes).type(F.int64)] = -1

        new_ntypes = (g.ntypes, g.ntypes)
        ret = DGLBlock(subgidx, new_ntypes, g.etypes)
        assert ret.is_unibipartite

    else:
       
        func = dgl._ffi.function.get_global_func('sampling.neighbor._CAPI_DGLSampleNeighbors2')

        subgidx = func(
#        subgidx = _CAPI_DGLSampleNeighbors(
            g._graph,
            nodes_all_types,
            fanout_array,
            edge_dir,
            prob_arrays,
            excluded_edges_all_t,
            replace,
        )
        ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)
        induced_edges = subgidx.induced_edges

    # handle features
    # (TODO) (BarclayII) DGL distributed fails with bus error, freezes, or other
    # incomprehensible errors with lazy feature copy.
    # So in distributed training context, we fall back to old behavior where we
    # only set the edge IDs.
    if not _dist_training:
        if copy_ndata:
            if fused:
                src_node_ids = [F.from_dgl_nd(src) for src in induced_nodes]
                dst_node_ids = [
                    utils.toindex(
                        nodes.get(ntype, []), g._idtype_str
                    ).tousertensor(ctx=F.to_backend_ctx(g._graph.ctx))
                    for ntype in g.ntypes
                ]
                node_frames = utils.extract_node_subframes_for_block(
                    g, src_node_ids, dst_node_ids
                )
                utils.set_new_frames(ret, node_frames=node_frames)
            else:
                node_frames = utils.extract_node_subframes(g, device)
                utils.set_new_frames(ret, node_frames=node_frames)

        if copy_edata:
            if fused:
                edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges]
                edge_frames = utils.extract_edge_subframes(g, edge_ids)
                utils.set_new_frames(ret, edge_frames=edge_frames)
            else:
                edge_frames = utils.extract_edge_subframes(g, induced_edges)
                utils.set_new_frames(ret, edge_frames=edge_frames)

    else:
        for i, etype in enumerate(ret.canonical_etypes):
            ret.edges[etype].data[EID] = induced_edges[i]

    return ret


def sample_neighbors2(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    _dist_training=False,
    exclude_edges=None,
    output_device=None,
):
    if F.device_type(g.device) == "cpu" and not g.is_pinned():
      #  print("CPU Sampling")
        frontier = _sample_neighbors2(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
            exclude_edges=exclude_edges,
        )
    else:
      #  print("GPU Sampling")
        frontier = _sample_neighbors2(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
        )
        if exclude_edges is not None:
            eid_excluder = EidExcluder(exclude_edges)
            frontier = eid_excluder(frontier)
    return frontier if output_device is None else frontier.to(output_device)



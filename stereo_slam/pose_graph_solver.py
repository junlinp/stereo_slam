import pypose as pp
import torch
import torch.nn as nn
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
import numpy as np
from codetiming import Timer

class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(nodes)

    def forward(self, edges:torch.Tensor, relative_poses):
        #torch_type = torch.get_default_dtype()

        # print(f"self.nodes: {self.nodes}")
        # print(f"edges: {edges}")
        # print(f"relative_poses: {relative_poses}")
        node1 = self.nodes[edges[..., 0]] #from Twi
        node2 = self.nodes[edges[..., 1]] #to Twj

        error = relative_poses @ node1.Inv() @ node2 # Tij^-1 * Tiw * Twj, pose = Tij
        first_pose = pp.SE3(np.array([0,0,0,0,0,0,1], dtype=np.float64)).to("cuda").to(torch.float64)
        prior_error = first_pose.Inv() @ self.nodes[0]
        return prior_error.Log().tensor(), error.Log().tensor()

class PoseGraphSolver:

    def __init__(self):
        info = np.eye(6)
        info[3:,3:] = info[3:,3:] * 40
        self.loop_info = torch.from_numpy(info).to("cuda").to(torch.float64)
        self.nodes = {}
        self.edges = []
        self.relative_poses = []

    def add_node(self, node_id: int, pose: pp.SE3):
        self.nodes[node_id] = pose

    def add_edge(self, node1: int, node2: int, relative_pose: pp.SE3):
        self.edges.append(np.array([node1, node2]))
        self.relative_poses.append(relative_pose)

    def get_optimized_pose(self, node_id: int) -> pp.SE3:
        return self.nodes[node_id]

    def get_all_optimized_poses(self):
        return self.nodes

    def optimize(self):

            id_to_index = {k: i for i, k in enumerate(self.nodes.keys())}
            nodes_array = torch.stack([self.nodes[k] for k in self.nodes.keys()]).to("cuda").to(torch.float64)
            edges_array = torch.tensor([np.array([id_to_index[edge[0]], id_to_index[edge[1]]]) for edge in self.edges], dtype=torch.int64, device="cuda")
            graph = PoseGraph(nodes_array).to("cuda")

            #print(f"self.loop_info: {self.loop_info}")
            pgo_infos = torch.stack([self.loop_info for _ in range(len(self.edges))])
            weight = (torch.eye(6,device='cuda') * 1e6, pgo_infos)

            with Timer(text="[posegraph] Elapsed time: {milliseconds:.0f} ms"):
                solver = ppos.Cholesky()
                strategy = ppost.TrustRegion(radius=1e4)
                kernel = (ppok.Scale().to("cuda"), ppok.Huber(delta = 0.1).to("cuda"))
                optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, kernel=kernel, vectorize=True)
                scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)
                while scheduler.continual():
                    loss = optimizer.step(input=(edges_array, torch.stack(self.relative_poses).to(torch.float64).to("cuda")), weight=weight)
                    scheduler.step(loss)
                    print(f'PyPose PGO at the {scheduler.steps} step(s) with loss {loss.item()}')

             
            self.nodes = {k: graph.nodes[id_to_index[k]].detach().cpu() for k in self.nodes.keys()}
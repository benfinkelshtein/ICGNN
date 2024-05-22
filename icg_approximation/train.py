from torch_geometric.data import Data
from torch.optim import Optimizer
from typing import Tuple

from icg_approximation.classes import DecompTrainArgs
from icg_approximation.utils import get_cut_norm
from helpers.constants import DECIMAL
from icg_approximation.loss import calc_efficient_graphon_loss
from icg_approximation.model import DecompModel


def train_decomp(data: Data, model: DecompModel, optimizer: Optimizer, train_args: DecompTrainArgs,
                 pbar, device, is_spatio_temporal: bool) -> Tuple[float, float, DecompModel]:
    model.train()

    best_loss, best_graphon_re = float('inf'), float('inf')
    best_model_state_dict = model.state_dict()
    for epoch in range(train_args.epochs):
        optimizer.zero_grad()
        loss, graphon_loss, graphon_re =\
            calc_efficient_graphon_loss(data=data, model=model, loss_scale=train_args.loss_scale, device=device,
                                        node_drop_ratio=train_args.node_drop_ratio,
                                        is_spatio_temporal=is_spatio_temporal)

        detach_loss = loss.item()
        graphon_re = graphon_re.item()
        loss.backward()
        optimizer.step()

        # best
        if detach_loss < best_loss:
            best_model_state_dict = model.state_dict()
            best_loss = detach_loss
            best_graphon_re = graphon_re

        # print
        log_str = f';GS;epoch:{epoch};loss={round(detach_loss, DECIMAL)}({round(best_loss, DECIMAL)})'
        log_str += f';g_re={round(graphon_re, DECIMAL)}({round(best_graphon_re, DECIMAL)})'
        if train_args.cut_norm:
            cutn_round, cutn_sdp = get_cut_norm(model=model, edge_index=data.edge_index)
            log_str += f';cutn_round={round(cutn_round, DECIMAL)};cutn_sdp={round(cutn_sdp, DECIMAL)}'
        pbar.set_description(log_str)
        pbar.update(n=1)

    model.eval()
    model.load_state_dict(best_model_state_dict)
    return best_loss, best_graphon_re, model

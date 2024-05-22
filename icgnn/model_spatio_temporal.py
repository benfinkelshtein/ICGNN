from typing import Union, List, Optional
from torch import Tensor
from torch_geometric.typing import Adj
from tsl.nn.blocks.encoders import RNN
from torch.nn import ModuleList

from icgnn_spatio_temporal.lib.nn.models.base.prototypes import TimeThenSpace

from icgnn.classes import CommArgs
from icg_approximation.model import DecompModel
from icgnn.model import CommModel


class TTSCommModel(TimeThenSpace):

    def __init__(self, model_args: CommArgs, icg_approx_model: DecompModel,
                 input_size: int, horizon: int, output_size: int,
                 exog_size: int = 0,
                 n_nodes: int = None,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        rnn = RNN(input_size=model_args.hidden_dim,
                  hidden_size=model_args.hidden_dim,
                  n_layers=1,
                  return_only_last_state=True,
                  cell='gru')
        mp_layers = ModuleList([CommModel(model_args=model_args, icg_approx_model=icg_approx_model)])
        super(TTSCommModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=rnn,
            spatial_encoder=mp_layers,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=model_args.hidden_dim,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # temporal encoding
        out = self.temporal_encoder(x)
        # spatial encoding
        for layer in self.mp_layers:
            out = layer(out, edge_index)
        return out

    def set_icg_approx_after_training(self):
        self.mp_layers[0].set_icg_approx_after_training()

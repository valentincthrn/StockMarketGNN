import torch

from src.utils.common import mape_loss


def run_all(
    data: dict,
    timestep: int,
    train_or_test: str,
    optimizer,
    lstms,
    my_gnn,
    mlp_heads,
    use_gnn,
):
    # Get the data
    data_t = data[train_or_test][timestep]
    pred_t = data["pred"][timestep]
    macro = data["macro"].get(timestep)

    if train_or_test == "train":
        optimizer.zero_grad()

    # PHASE 1: LSTM EXTRACTION
    features_extracted, comps = run_lstm_separatly(lstms, data_t)

    # PHASE 2: GNN EXTRACTION
    if use_gnn:
        features_encoded = my_gnn(features_extracted)
    else:
        features_encoded = features_extracted

    # PHASE 3: MLP HEAD EXTRACTION
    pred, true = run_mlp_heads_separatly(
        mlp_heads, features_encoded, comps, pred_t, macro
    )

    # Compute the loss
    loss = mape_loss(pred, true)

    if train_or_test == "train":
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return loss, pred, true, comps


def run_lstm_separatly(lstm_dict: torch.nn.ModuleDict, data_t: dict) -> torch.tensor:
    out_lstm = []
    comps = []

    # Run each LSTM separatly and aggregate the result in the same matrix
    for comp, tensor in data_t.items():
        tensor = tensor.unsqueeze(1)

        out_comp_lstm = lstm_dict[comp](tensor)

        # Getting the last output from LSTM
        out_lstm.append(out_comp_lstm.squeeze(0))
        comps.append(comp)

    # Concatenate the outputs
    features_extracted = torch.stack(out_lstm, dim=0).squeeze(1)

    return features_extracted, comps


def run_mlp_heads_separatly(
    mlp_heads: torch.nn.ModuleDict,
    features_encoded: torch.tensor,
    comps: list,
    pred_t: dict,
    macro: torch.tensor,
) -> torch.tensor:
    # Run each MLP separatly
    price_outputs_time_t = []
    pred_output_time_t = []

    for k, comp in enumerate(comps):
        out_gnn_comp_i = features_encoded[k]
        if macro is None:
            gnn_with_macro = out_gnn_comp_i
        else:
            gnn_with_macro = torch.concatenate([out_gnn_comp_i, macro])

        price_comp_i = mlp_heads[comp](gnn_with_macro)
        price_outputs_time_t.append(price_comp_i)
        pred_output_time_t.append(pred_t[comp])

    # Concatenate the outputs frm the LSTM
    pred = torch.stack(price_outputs_time_t, dim=0)

    # Prepare ground truth from d_pred for the current timestep
    true = torch.tensor(pred_output_time_t).reshape_as(pred).float()

    return pred, true

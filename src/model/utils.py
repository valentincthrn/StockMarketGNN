import torch

from src.utils.common import mape_loss, index_agreement_torch


def run_all(
    data: dict,
    timestep: int,
    train_or_test: str,
    optimizer,
    params,
    criterion: str,
    lstms,
    my_gnn,
    mlp_heads,
    use_gnn,
    device,
):
    # Get the data
    data_t = data[train_or_test][timestep]
    pred_t = data["pred"][timestep]
    macro = data["macro"].get(timestep)

    if train_or_test == "train":
        optimizer.zero_grad()

    # PHASE 1: LSTM EXTRACTION
    features_extracted, comps = run_lstm_separatly(lstms, data_t, device)

    # PHASE 2: GNN EXTRACTION
    if use_gnn:
        features_encoded = my_gnn(features_extracted)
    else:
        features_encoded = features_extracted

    # PHASE 3: MLP HEAD EXTRACTION
    pred, true = run_mlp_heads_separatly(
        mlp_heads, features_encoded, comps, pred_t, macro, device
    )

    # Compute the loss
    if criterion == "MAPE":
        loss = mape_loss(pred, true)
    elif criterion == "IoA":
        loss = index_agreement_torch(pred, true)

    if train_or_test == "train":
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_value_(params, 1.0)
        optimizer.step()

    return loss, pred, true, comps


def run_lstm_separatly(
    lstm_dict: torch.nn.ModuleDict, data_t: dict, device: str
) -> torch.tensor:
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
    device: str,
    to_pred: bool = False,
) -> torch.tensor:
    # Run each MLP separatly
    price_outputs_time_t = []
    pred_output_time_t = []

    for k, comp in enumerate(comps):
        out_gnn_comp_i = features_encoded[k].to(device)

        if macro is None:
            gnn_with_macro = out_gnn_comp_i
        else:
            gnn_with_macro = torch.cat([out_gnn_comp_i, macro])

        price_comp_i = mlp_heads[comp](gnn_with_macro)
        price_outputs_time_t.append(price_comp_i)

        if not to_pred:
            pred_output_time_t.append(pred_t[comp])

    # Concatenate the outputs frm the LSTM
    pred = torch.stack(price_outputs_time_t, dim=0)

    if to_pred:
        return pred

    # Prepare ground truth from d_pred for the current timestep
    true = torch.tensor(pred_output_time_t).reshape_as(pred).float().to(device)

    return pred, true

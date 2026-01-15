import torch
import support_src.model as support_model


def build_casupnet_from_L1(model_file, device):
    model = support_model.SUPPORT(
        in_channels=61,
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=1,
        bp=False,
    )

    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state, strict=False)  # như đã thống nhất
    model.to(device)
    model.eval()
    return model


def run_casupnet_on_stack(model, noisy_stack, device):
    """
    noisy_stack: torch.Tensor, shape [T, H, W] hoặc [1, T, H, W]
    trả về: torch.Tensor [H, W]
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        if noisy_stack.dim() == 3:
            noisy_stack = noisy_stack.unsqueeze(0)  # [1, T, H, W]
        noisy_stack = noisy_stack.to(device)
        pred = model(noisy_stack)  # [1, 1, H, W]
    return pred.squeeze(0).squeeze(0).cpu()

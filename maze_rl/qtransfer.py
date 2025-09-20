
import torch
import torch.nn.functional as F  # noqa: F401  (por si se quiere usar en el futuro)

def resize_Q(Q_prev, old_shape, new_shape):
    """Redimensiona un mapa Q (H*W, A) al nuevo tamaño (H'*W', A) con bilinear 2D."""
    if Q_prev is None:
        return None
    if old_shape == new_shape:
        return Q_prev.clone()
    old_h, old_w = old_shape; new_h, new_w = new_shape
    A = Q_prev.shape[1]
    grid = Q_prev.view(old_h, old_w, A).permute(2,0,1).unsqueeze(0)   # [1,A,H,W]
    grid2 = torch.nn.functional.interpolate(grid, size=(new_h, new_w), mode="bilinear", align_corners=False)
    Q_new = grid2.squeeze(0).permute(1,2,0).reshape(new_h*new_w, A).contiguous()
    return Q_new

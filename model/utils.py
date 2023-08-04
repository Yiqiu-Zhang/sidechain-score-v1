import torch
from typing import List
import torch.nn.functional as func


""" IPA utils functions"""
def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def permute_final_dims(tensor: torch.Tensor, idxs: List[int]):
    zero_index = -1 * len(idxs)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in idxs])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


"""Sturcture utils functions"""

def rbf(D, D_min = 0., D_max=20., D_count=16):
    # Distance radial basis function

    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1] * len(D.shape) + [-1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

    return RBF

def quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
    """
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(func.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = func.normalize(Q, dim=-1)

    # Axis of rotation
    # Replace bad rotation matrices with identity
    # I = torch.eye(3).view((1,1,1,3,3))
    # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
    # det = (
    #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
    #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
    #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
    # )
    # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
    # R = det_mask * R + (1 - det_mask) * I

    # DEBUG
    # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    # Columns of this are in rotation plane
    # A = R - I
    # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
    # axis = func.normalize(torch.cross(v1, v2), dim=-1)
    return Q

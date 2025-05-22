import torch
import torch.nn as nn
from torch.func import vmap
import numpy as np
import math
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def simple_uniform_init(shape, std=1., device=None, dtype=torch.float32):
    weight = torch.rand(shape, device=device, dtype=dtype)*2*std - std
    return weight


class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GLU, self).__init__()
        self.l1 = nn.Linear(input_size, output_size, bias=True)
        self.l2 = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.l1(x) * self.l2(x).sigmoid()


# Binary operator for parallel scan of linear recurrence.
def binary_operator(x_i, x_j):
    A_i, b_i = x_i
    A_j, b_j = x_j

    N = A_i.size(0) // 4
    iA_ = A_i[0 * N: 1 * N]
    iB_ = A_i[1 * N: 2 * N]
    iC_ = A_i[2 * N: 3 * N]
    iD_ = A_i[3 * N: 4 * N]
    jA_ = A_j[0 * N: 1 * N]
    jB_ = A_j[1 * N: 2 * N]
    jC_ = A_j[2 * N: 3 * N]
    jD_ = A_j[3 * N: 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_

    Anew = torch.cat([A_new, B_new, C_new, D_new], dim=0)

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = torch.cat([new_b1, new_b2], dim=0)

    return Anew, new_b + b_j


# only available on versions after Pytorch 2.0
batch_binary_operator = vmap(binary_operator)


# Associative Scan to do same thing as JAX associative scan
def associative_scan(fn, inputs):
    xs, ys = inputs
    out_x, out_y = [], []
    acc_x, acc_y = xs[0], ys[0]
    out_x.append(acc_x)
    out_y.append(acc_y)

    for i in range(1, xs.shape[0]):
        acc_x, acc_y = fn((acc_x, acc_y), (xs[i], ys[i]))
        out_x.append(acc_x)
        out_y.append(acc_y)

    return torch.stack(out_x), torch.stack(out_y)


# Pytorch version -- Implicit Method
def apply_linoss_im(A_diag, B, C_til, input_seq, step):
    pass
    """
    Args...
    :param A_diag: Tensor(P,)
    :param B: Tensor (P, H)
    :param C_til: Tensor (H,P)
    :param input_seq: Tensor(L, H)
    :param step: Tensor (P,)
    :return: (L, H)
    """
    device = input_seq.device
    L, H = input_seq.shape
    P = A_diag.shape[0]

    Bu_elements = torch.matmul(input_seq.to(torch.complex64), B.T)

    # Implicit Method computation
    schur_comp = 1. / (1. + step**2 * A_diag)
    M_11 = 1 - step**2 * A_diag * schur_comp
    M_12 = - step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = torch.cat([M_11, M_12, M_21, M_22], dim=0)
    # M ---> (4P, )

    M_elements = M.unsqueeze(0).repeat(L, 1)
    # M_elements ----> (L, 4P)

    # Affine Part
    F1 = step * (M_11.unsqueeze(0)) * Bu_elements
    F2 = step * (M_21.unsqueeze(0)) * Bu_elements
    F = torch.cat([F1, F2], dim=1)
    # F ----> (L, 2P)

    # Associative Scan
    _, xs = associative_scan(binary_operator, (M_elements, F))
    # xs ---> (L , 2P)

    # extract ----> (L, P)
    ys = xs[:, P:]

    # output ---> (L, P) x (P, H) = (L, H)
    output = torch.matmul(ys, C_til.T).real

    return output


# Implicit-Explicit Method
def apply_linoss_imex(A_diag, B, C, input_sequence, step):
    pass
    """
    Args....
    :param A_diag: Tensor (P, )
    :param B: Tensor (P, H)
    :param C: Tensor (H, P)
    :param input_sequence: (L, H)
    :param step: (P, )
    :return (L,H)
    """
    device = input_sequence.device
    L, H = input_sequence.shape
    P = A_diag.shape[0]

    Bu_elements = torch.matmul(input_sequence.to(torch.complex64), B.T)

    # Implicit Explicit Method computation
    A_ = torch.ones_like(A_diag, dtype=torch.complex64)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M = torch.cat([A_, B_, C_, D_], dim=0)
    # M ----> (4P, )

    M_elements = M.unsqueeze(0).repeat(L, 1)
    # M_elements ----> (L, 4P)

    # Affine Part
    F1 = Bu_elements * step
    F2 =  Bu_elements * step
    F = torch.cat([F1, F2], dim=1)

    # associative scan
    _, xs = associative_scan(binary_operator, (M_elements, F))
    # xs ----> (L, 2P)

    # Extract : (L, P)
    ys = xs[:, P:]

    # output : (L, H) = (L, P) x (P, H)
    output = torch.matmul(ys, C.T).real

    return output


class LinOSSLayer(nn.Module):
    def __init__(self, ssm_size, H, discretization, device=None):
        super().__init__()
        self.A_diag = nn.Parameter(torch.rand(ssm_size, device=device))
        B_real = simple_uniform_init((ssm_size, H), std=1./math.sqrt(H), device=device)
        B_img = simple_uniform_init((ssm_size, H), std=1./math.sqrt(H), device=device)
        self.B_real = nn.Parameter(B_real)
        self.B_img = nn.Parameter(B_img)

        C_real = simple_uniform_init((H, ssm_size), std=1./math.sqrt(ssm_size), device=device)
        C_img = simple_uniform_init((H, ssm_size), std=1./math.sqrt(ssm_size), device=device)
        self.C_real = nn.Parameter(C_real)
        self.C_img = nn.Parameter(C_img)

        self.D = nn.Parameter(torch.rand(H))

        self.steps = nn.Parameter(torch.rand(ssm_size, device=device))
        self.discretization = discretization

    def forward(self, input_sequence):
        A_diag = F.relu(self.A_diag)

        B = torch.complex(self.B_real, self.B_img)
        C = torch.complex(self.C_real, self.C_img)
        
        steps = torch.sigmoid(self.steps)

        # discretization dispatch
        if self.discretization == 'IMEX':
            ys = apply_linoss_imex(A_diag, B, C, input_sequence, steps)
        elif self.discretization == 'IM':
            ys = apply_linoss_im(A_diag, B, C, input_sequence, steps)
        else:
            raise NotImplementedError(f"Discretization '{self.discretization}' not supported.")

        Du = input_sequence * self.D

        return ys + Du



# H --> hidden layer dimension
class LinOSSBlock(nn.Module):
    def __init__(self, ssm_size, H, discretization, drop_rate=0.05):
        super().__init__()
        self.norm = nn.BatchNorm1d(H, affine=False)
        self.ssm = LinOSSLayer(ssm_size, H, discretization)
        self.glu = GLU(H, H)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        """

        :param x: (batch_dim, seq_len, hidden_dim)
        """
        skip = x
        # Applying BatchNorm
        x = x.transpose(1, 2)   # (B, H, T)
        x = self.norm(x)
        x = x.transpose(1, 2)   # (B, T, H)

        # Main...
        x = torch.stack([self.ssm(xi) for xi in x], dim=0)
        x = self.drop(F.gelu(x))
        x = self.glu(x)
        x = self.drop(x)

        x = x + skip

        return x


# N ---> input dimension
# H ---> hidden layer dimension
# O ---> output dimension
class LinOSS(nn.Module):
    def __init__(self, num_block, N, ssm_size, H, O, classification, output_step, discretization, drop_rate=0.05):
        super().__init__()
        self.l1 = nn.Linear(N, H)
        self.blocks = nn.ModuleList([
            LinOSSBlock(ssm_size, H, discretization, drop_rate)
            for _ in range(num_block)
        ])
        self.l2 = nn.Linear(H, O)
        self.classification = classification
        self.output_step = output_step

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim) if classification,
                         or (batch_size, num_outputs, output_dim) otherwise
        """
        batch_size, seq_len = x.size

        x = self.l1(x)

        for block in self.blocks:
            x = block(x)

        if self.classification:
            # (B, T, H) --> (B, H)
            x = x.mean(dim=1)
            x = self.l2(x)
            x = F.softmax(x, dim=-1)
        else:
            x = x[:, self.output_step - 1 :: self.output_step, :]
            x = self.l2(x)
            x = torch.tanh(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 50
    hidden_dim = 64
    ssm_size = 128

    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Instantiate LinOSSBlock
    block = LinOSSBlock(
        ssm_size=ssm_size,
        H=hidden_dim,
        discretization="IM",
        drop_rate=0.1
    )

    # Run forward pass
    output = block(x)

    # Print outputs
    print("Output shape:", output.shape)
    print("Output sample:", output[0, :3])








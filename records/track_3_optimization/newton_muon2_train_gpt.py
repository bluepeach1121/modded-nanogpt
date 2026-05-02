"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)


@torch.no_grad()
def accum_xtx(x: Tensor, accum: Tensor, count: Tensor):
    x2d = x.detach().float().reshape(-1, x.size(-1))
    K = (x2d.T @ x2d) / x2d.size(0)
    accum.add_(K)
    count.add_(1.0)

@torch.no_grad()
def accum_xtx_blocks4(x: Tensor, accum: Tensor, count: Tensor):
    x2d = x.detach().float().reshape(-1, x.size(-1))
    N, fourD = x2d.shape
    assert fourD % 4 == 0
    D = fourD // 4
    z = x2d.view(N, 4, D)
    K = torch.einsum("nbi,nbj->bij", z, z) / N
    accum.add_(K)
    count.add_(1.0)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

        d = dim
        self.register_buffer("qkv_xtx_accum", torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.register_buffer("qkv_xtx_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("o_xtx_accum", torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.register_buffer("o_xtx_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self._refresh_stats_refs()

    def _refresh_stats_refs(self):
        d = self.proj.weight.size(1)
        qkv_ref = {"kind": "qkv", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.q.weight._stats_ref = qkv_ref
        self.k.weight._stats_ref = qkv_ref
        self.v.weight._stats_ref = qkv_ref
        self.proj.weight._stats_ref = {"kind": "o", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}

    def _apply(self, fn):
        super()._apply(fn)
        self._refresh_stats_refs()
        return self

    def forward(self, x: Tensor, precond_flag: bool = False):
        B, T = x.size(0), x.size(1)
        if precond_flag:
            accum_xtx(x, self.qkv_xtx_accum, self.qkv_xtx_count)

        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)

        if precond_flag:
            accum_xtx(y, self.o_xtx_accum, self.o_xtx_count)

        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

        d = dim
        self.register_buffer("fc_xtx_accum", torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.register_buffer("fc_xtx_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("proj_xtx_accum", torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
        self.register_buffer("proj_xtx_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self._refresh_stats_refs()

    def _refresh_stats_refs(self):
        d = self.fc.weight.size(1)
        self.fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
        self.proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}

    def _apply(self, fn):
        super()._apply(fn)
        self._refresh_stats_refs()
        return self

    def forward(self, x: Tensor, precond_flag: bool = False):
        if precond_flag:
            accum_xtx(x, self.fc_xtx_accum, self.fc_xtx_count)

        x = self.fc(x)
        x = x.relu().square()

        if precond_flag:
            accum_xtx_blocks4(x, self.proj_xtx_accum, self.proj_xtx_count)

        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor, precond_flag: bool = False):
        x = x + self.attn(self.norm1(x), precond_flag=precond_flag)
        x = x + self.mlp(self.norm2(x), precond_flag=precond_flag)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor, precond_flag: bool = False):
        precond_flag = bool(precond_flag) and self.training
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x, precond_flag=precond_flag)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def newton_muon2_update(grad, momentum, velocity, mu=0.95, beta2=0.9, eps=1e-8, nesterov=True):
    # grad is already activation-right-preconditioned by Newton-Muon.
    # Muon² adds Adam-style second-moment scaling before Newton-Schulz.
    momentum.lerp_(grad, 1 - mu)
    velocity.lerp_(grad.square(), 1 - beta2)

    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = update / (velocity.sqrt() + eps)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class NewtonMuon2(torch.optim.Optimizer):
    """
    Track-3-shaped Newton-Muon².

    This preserves train_gpt_simple.py's architecture, data path, LR schedule, and
    distributed Muon update pattern while adding the Newton-Muon right preconditioner:
        raw grad -> grad @ inv(E[z z^T] + ridge) -> Muon momentum + second moment -> Newton-Schulz.

    Activation second moments are collected only on refresh steps. The MLP projection
    uses four block-diagonal dxd preconditioners for its 4d input, matching the
    Newton-Muon repository implementation for contraction matrices.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, beta2=0.9, eps=1e-8,
                 precond_beta=0.95, precond_ridge_mult=0.2,
                 precond_init_diag=1e-3, precond_eps=1e-8,
                 refresh_interval=32):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, beta2=beta2, eps=eps)
        super().__init__(params, defaults)
        self.precond_beta = float(precond_beta)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_init_diag = float(precond_init_diag)
        self.precond_eps = float(precond_eps)
        self.refresh_interval = int(refresh_interval)
        self.global_step = 0
        self._precond_ready = False

    def precond_flag_for_step(self, step: int) -> bool:
        # Faithful to the repo: t = step + 1, so first capture/refresh is step 31 for interval=32.
        return ((int(step) + 1) % self.refresh_interval) == 0

    def _iter_params_with_stats(self):
        for group in self.param_groups:
            for p in group["params"]:
                stref = getattr(p, "_stats_ref", None)
                if stref is not None:
                    yield p, stref

    def _init_precond_state(self, p: Tensor, stref: dict):
        state = self.state[p]
        if "precond_kind" in state:
            return

        kind = stref["kind"]
        d = int(stref["d"])
        state["precond_kind"] = kind
        state["precond_d"] = d

        if kind in ("qkv", "o", "c_fc"):
            cov = torch.zeros((d, d), device=p.device, dtype=torch.float32)
            cov.diagonal().fill_(self.precond_init_diag)
            inv = torch.eye(d, device=p.device, dtype=torch.float32)
        elif kind == "c_proj":
            cov = torch.zeros((4, d, d), device=p.device, dtype=torch.float32)
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
            inv = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            inv.zero_()
            inv.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        else:
            raise ValueError(f"unknown Newton-Muon preconditioner kind: {kind}")

        state["precond_cov"] = cov
        state["precond_inv"] = inv

    @torch.no_grad()
    def _inverse_with_ridge(self, cov: Tensor) -> Tensor:
        K = cov.float().clone()
        K = 0.5 * (K + K.transpose(-1, -2))
        diag = K.diagonal(dim1=-2, dim2=-1)
        ridge = (diag.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps).clamp_min(self.precond_eps)
        diag.add_(ridge.unsqueeze(-1))

        L, info = torch.linalg.cholesky_ex(K, upper=False, check_errors=False)
        inv = torch.cholesky_inverse(L, upper=False)

        # If Cholesky fails for any matrix, fall back to identity for that matrix.
        if info.numel() == 1:
            if int(info.item()) != 0:
                inv.zero_()
                inv.diagonal().fill_(1.0)
        else:
            bad = info != 0
            if bad.any():
                inv[bad].zero_()
                inv[bad].diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return inv

    @torch.no_grad()
    def _refresh_preconditioners(self):
        # Every rank iterates all stats in the same order, so distributed all_reduce is aligned.
        seen_stats = []
        for p, stref in self._iter_params_with_stats():
            self._init_precond_state(p, stref)
            state = self.state[p]
            count = stref["count"].clamp_min(1.0)
            K_batch = stref["accum"] / count
            if dist.is_initialized():
                dist.all_reduce(K_batch, op=dist.ReduceOp.SUM)
                K_batch /= dist.get_world_size()

            has_stats = bool((stref["count"] > 0).item())
            if has_stats:
                state["precond_cov"].lerp_(K_batch, 1.0 - self.precond_beta)
                state["precond_inv"] = self._inverse_with_ridge(state["precond_cov"])
            seen_stats.append(stref)

        # Reset each unique stats buffer once after all params using it have been updated.
        reset_ids = set()
        for stref in seen_stats:
            key = id(stref["accum"])
            if key not in reset_ids:
                stref["accum"].zero_()
                stref["count"].zero_()
                reset_ids.add(key)
        self._precond_ready = True

    @torch.no_grad()
    def _right_precondition_grad(self, grad: Tensor, state: dict) -> Tensor:
        if not self._precond_ready or "precond_inv" not in state:
            return grad

        inv = state["precond_inv"]
        if state.get("precond_kind") == "c_proj":
            # grad shape is [d, 4d]. Apply four separate dxd right preconditioners.
            chunks = torch.chunk(grad.float(), 4, dim=1)
            out = [chunks[i] @ inv[i] for i in range(4)]
            return torch.cat(out, dim=1).type_as(grad)
        return (grad.float() @ inv).type_as(grad)

    @torch.no_grad()
    def step(self):
        do_refresh = self.precond_flag_for_step(self.global_step)
        if do_refresh:
            self._refresh_preconditioners()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                        state["velocity"] = torch.zeros_like(p)

                    g = self._right_precondition_grad(p.grad, state)
                    update = newton_muon2_update(
                        g,
                        state["momentum"],
                        state["velocity"],
                        mu=group["mu"],
                        beta2=group["beta2"],
                        eps=group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        self.global_step += 1


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = 3500

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = NewtonMuon2(
    [p for p in model.blocks.parameters() if p.ndim >= 2],
    lr=0.0375,
    weight_decay=0.025,
    mu=0.95,
    beta2=0.9,
    eps=1e-8,
    precond_beta=0.95,
    precond_ridge_mult=0.2,
    precond_init_diag=1e-3,
    precond_eps=1e-8,
    refresh_interval=32,
)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % 125 == 0:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs], precond_flag=False)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    optimizer2.global_step = step
    precond_flag = optimizer2.precond_flag_for_step(step)
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs], precond_flag=precond_flag).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    ########### additionbelow
    for name, p in model.named_parameters():
        if not torch.isfinite(p.grad).all():
            print0(f"NONFINITE GRAD at step {step}: {name}", console=True)
            raise RuntimeError("nonfinite grad")
    ########## additionabove
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    ######### additionbelow
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print0(f"NONFINITE PARAM at step {step}: {name}", console=True)
            raise RuntimeError("nonfinite param")
    #########additionabove
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()

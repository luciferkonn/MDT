'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-02-06 18:30:27
LastEditors: Jikun Kang
FilePath: /MDT/src/relational_memory.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupLinearLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_blocks,
            bias=True,
            a=None,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        if a is None:
            a = 1. / math.sqrt(out_dim)
        self.weight = nn.Parameter(torch.FloatTensor(
            num_blocks, in_dim, out_dim).uniform_(-a, a))
        self.bias = bias
        if self.bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                num_blocks, out_dim).uniform_(-a, a))
        else:
            self.bias = None

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)
        if self.bias is not None:
            x = x + self.bias

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=300) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pe = torch.zeros(max_seq_len, hidden_dim)
        for pos in range(max_seq_len):
            for i in range(0, hidden_dim, 2):
                pe[pos, i] = math.sin(pos/(10, 000**((2*i)/hidden_dim)))
                pe[pos, i+1] = math.cos(pos/(10, 000**((2*(i+1))/hidden_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.pos_embed_weight = nn.Parameter(torch.ones_like(pe))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        seq_len = x.size(1)

        # (bs,pos,nhid) * (bs, nhid, pos) = (bs, pos, nhid)
        pe_use = self.pe[:, :seq_len] * \
            F.sigmoid(self.pos_embed_weight[:, :seq_len])
        x = x+pe_use
        x = x.permute(1, 0, 2)
        return x


class RepeatLinear(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_steps,
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoder(in_dim)
        self.num_steps = num_steps
        self.w = nn.Parameter(torch.randn(in_dim).cuda())
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        w = self.w.unsqueeze(0).repeat(self.num_steps, 1)
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)

        x = torch.relu(w * x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)

        return x


class RelationalMemory(nn.Module):
    def __init__(
            self,
            mem_slots,
            head_size,
            input_size,
            num_heads: int = 1,
            num_blocks: int = 1,
            forget_bias=1.,
            input_bias=0.,
            gate_style="unit",
            attention_mlp_layers=2,
            key_size=None,
            return_all_outputs=False,
            use_topk=False,
            topk: int = 3,
            num_steps: int = 5,
            null_attention=False,
    ) -> None:
        super().__init__()

        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads
        self.use_topk = use_topk
        self.topk = topk

        self.mem_slots_plus_input = self.mem_slots + 1

        assert num_blocks < 1 (f"num blocks mush be >= 1. Got: {num_blocks}")

        self.num_blocks = num_blocks
        self.gate_style = gate_style

        self.num_atten_mlp_layers = attention_mlp_layers
        self.key_size = key_size if key_size else self.head_size

        # value size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_value = 2*self.key_size + self.value_size
        self.total_qkv_size = self.qkv_value*self.num_heads

        self.query_proj = nn.Linear(
            self.mem_size, self.key_size*self.num_heads)
        count_parameters(self.query_proj, "query")
        self.key_proj = nn.Linear(self.mem_size, self.key_size*self.num_heads)
        count_parameters(self.key_proj, "key")
        self.value_proj = nn.Linear(
            self.mem_size, self.value_size*self.num_heads)
        count_parameters(self.value_proj, "value")

        self.attention_mlp = nn.ModuleList(
            [nn.Linear(self.mem_size, self.mem_size)]*self.num_atten_mlp_layers)
        count_parameters(self.attention_mlp[0], "attention_mlp")
        self.attended_layer_norm = nn.LayerNorm(self.mem_size)
        count_parameters(self.attended_layer_norm, "layer_norm1")
        self.attended_layer_norm2 = nn.LayerNorm(self.mem_size)
        count_parameters(self.attended_layer_norm2, "layer_norm2")

        # params for initial embedding function
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)
        count_parameters(self.input_projector, "input_projector")

        # params for gating
        self.num_gates = 2 * self.calculate_gate_size()
        print("input projector:"+str(self.mem_size))
        if gate_style in ['unit', 'memory']:
            self.input_gate_projector = RepeatLinear(
                in_dim=self.mem_size, out_dim=self.num_gates, num_steps=num_steps)
            count_parameters(self.input_gate_projector, "input_gate_projector")
            self.memory_gate_projector = GroupLinearLayer(
                in_dim=self.mem_size, out_dim=num_blocks, num_blocks=self.num_gates)
            count_parameters(self.memory_gate_projector, "memory_gate_projector")
        
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))

    def calculate_gate_size(self):
        if self.gate_style == "unit":
            return self.mem_size
        elif self.gate_style == "memory":
            return 1
        else:
            return 0


def count_parameters(model, name):
    k = 0
    for p in model.parameters():
        k += p.numel()
    print(name, end=':')
    print(k)

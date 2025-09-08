import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Attention, GPT2Block, GPT2Config

# n_dim is the dimension of feature X_i (ith column of X)
# Custom GPT2Attention to replace softmax with ReLU
class GPT2Attention_relu(GPT2Attention):
    def __init__(self, config, is_relu=False):
        super().__init__(config)
        self.is_relu = is_relu

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if not self.is_cross_attention:
        #     # if only "normal" attention layer implements causal mask
        #     query_length, key_length = query.size(-2), key.size(-2)
        #     causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        #     mask_value = torch.finfo(attn_weights.dtype).min
        #     # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        #     # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        #     mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        #     attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        if self.is_relu:
            attn_weights = nn.functional.relu(attn_weights)
            # print("Using relu function.")
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)


        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

# Custom GPT2Block to remove LayerNorm
class CustomGPT2Block(GPT2Block):
    def __init__(self, config, is_relu=False, is_layernorm=True):
        super().__init__(config)
        # Replace the attention with our custom attention
        self.attn = GPT2Attention_relu(config, is_relu=is_relu)
        if not is_layernorm:
            # print("Remove layer normalization.")
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
# Custom GPT2 model to use the modified blocks
class CustomGPT2Model(GPT2Model):
    def __init__(self, config, is_relu, is_layernorm):
        super().__init__(config)
        # Replace all layers with our custom block
        self.h = nn.ModuleList([CustomGPT2Block(config, is_relu=is_relu, is_layernorm=is_layernorm) for _ in range(config.n_layer)])




class TransformerModel(nn.Module):
    def __init__(self, n_dims, N, n_positions, n_embd, n_layer, n_head, input_is_cov, predict_vector, predict_cov,is_relu, is_layernorm, k=1):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = CustomGPT2Model(configuration, is_relu = is_relu, is_layernorm = is_layernorm)
        self.predict_cov = predict_cov
        if input_is_cov:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*n_dims, n_dims)
            else:
                self._read_out = nn.Linear(n_embd*n_dims, k)
        else:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*N, n_dims*k)
            else:
                self._read_out = nn.Linear(n_embd*N, k)
        if predict_cov:
            self._read_out = nn.Linear(n_embd * N,n_dims * n_dims)


    def forward(self, xs, inds=None):

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        # print(output.shape)
        flatten_output = torch.flatten(output, start_dim=1)
        # print(flatten_output.shape)
        prediction = self._read_out(flatten_output)
        # print(prediction.shape)
        if self.predict_cov:
            prediction = prediction.view(-1, self.n_dims, self.n_dims)
        return prediction



class TransformerModel_2(nn.Module):
    def __init__(self, n_dims, N, n_positions, n_embd, n_layer, n_head, input_is_cov, predict_vector, predict_cov,is_relu, is_layernorm, k=1):
        super(TransformerModel_2, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = CustomGPT2Model(configuration, is_relu = is_relu, is_layernorm = is_layernorm)
        self.predict_cov = predict_cov
        if input_is_cov:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*n_dims, n_dims)
            else:
                self._read_out = nn.Linear(n_embd*n_dims, k)
        else:
            if predict_vector:
                # self._read_out = nn.Linear(n_embd*N, n_dims*k)
                self._read_out = nn.Linear(n_embd, k*n_dims)
            else:
                self._read_out = nn.Linear(n_embd*N, k)
        if predict_cov:
            self._read_out = nn.Linear(n_embd * N,n_dims * n_dims)


    def forward(self, xs, inds=None):

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        # print(output.shape)
        # flatten_output = torch.flatten(output, start_dim=1)
        # print(flatten_output.shape)
        prediction = self._read_out(output[:,0,:])
        # print(prediction.shape)
        if self.predict_cov:
            prediction = prediction.view(-1, self.n_dims, self.n_dims)
        return prediction
# test
# model = TransformerModel(D, N+10, n_embd=128, n_layer=6, n_head=4)
# output = model(training_data_tensor)
# print(output.shape)

class TransformerModel_drop(nn.Module):
    def __init__(self, n_dims, N, n_positions, n_embd, n_layer, n_head, input_is_cov, predict_vector, predict_cov,is_relu, is_layernorm, k=1):
        super(TransformerModel_drop, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = CustomGPT2Model(configuration, is_relu = is_relu, is_layernorm = is_layernorm)
        self.predict_cov = predict_cov
        if input_is_cov:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*n_dims, n_dims)
            else:
                self._read_out = nn.Linear(n_embd*n_dims, k)
        else:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*N, n_dims*k)
            else:
                self._read_out = nn.Linear(n_embd*N, k)
        if predict_cov:
            self._read_out = nn.Linear(n_embd * N,n_dims * n_dims)


    def forward(self, xs, inds=None):

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        # print(output.shape)
        flatten_output = torch.flatten(output, start_dim=1)
        # print(flatten_output.shape)
        prediction = self._read_out(flatten_output)
        # print(prediction.shape)
        if self.predict_cov:
            prediction = prediction.view(-1, self.n_dims, self.n_dims)
        return prediction

import torch.nn.functional as F


def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError


class EncoderTransformer(nn.Module):
    def __init__(self, n_dims, N, input_is_cov, predict_vector, predict_cov, k=1, normalize_attn=False, mlp=True, layernorm=True, n_embd=64, n_layer=3, n_head=4, activation="relu"):
        super(EncoderTransformer, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        # self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        # layers
        self._read_in = nn.Linear(n_dims, n_embd)
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_embd),
                    nn.ReLU(),
                    nn.Linear(n_embd, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))
        # self._read_out = nn.Linear(n_embd, n_class)
        self.predict_cov = predict_cov
        if input_is_cov:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*n_dims, n_dims)
            else:
                self._read_out = nn.Linear(n_embd*n_dims, k)
        else:
            if predict_vector:
                self._read_out = nn.Linear(n_embd*N, n_dims*k)
            else:
                self._read_out = nn.Linear(n_embd*N, k)
        if predict_cov:
            self._read_out = nn.Linear(n_embd * N,n_dims * n_dims)
    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Directly stack the x's and y's into the same location
        resulting sequence would be Bx(N+1)x(d+1), where (N+1)-th token is test
        """
        zs = torch.cat((xs_b, ys_b.unsqueeze(2)), dim=2)
        zs[:, -1, -1].zero_()
        return zs

    def forward(self, xs):
        H = self._read_in(xs)
        for (q, k, v, ln1, mlp, ln2) in zip(
            self._queries, self._keys, self._values,
            self._lns_1, self._mlps, self._lns_2,
        ):
            query = q(H)
            key = k(H)
            value = v(H)

            attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key))
            if self.normalize_attn:
                attn_weights = attn_weights / xs.shape[1]
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value)
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)
        # print("Encoder output shape: ", H.shape)
        # prediction = self._read_out(H)
        # print("Encoder output shape: ", prediction.shape)
        flatten_output = torch.flatten(H, start_dim=1)
        # print(flatten_output.shape)
        prediction = self._read_out(flatten_output)
        # print(prediction.shape)
        if self.predict_cov:
            prediction = prediction.view(-1, self.n_dims, self.n_dims)
        return prediction


    # def forward_with_attn(self, xs):
    #     H = self._read_in(xs)
    #     for (q, k, v, ln1, mlp, ln2) in zip(
    #         self._queries, self._keys, self._values,
    #         self._lns_1, self._mlps, self._lns_2,
    #     ):
    #         query = q(H)
    #         key = k(H)
    #         value = v(H)

    #         attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key))
    #         if self.normalize_attn:
    #             attn_weights = attn_weights / xs.shape[1]
    #         H = H + torch.einsum('bij,bjd->bid', attn_weights, value)
    #         if self.layernorm:
    #             H = ln1(H)

    #         if self.mlp:
    #             H = H + mlp(H)
    #             if self.layernorm:
    #                 H = ln2(H)
    #     print("Encoder output shape: ", H.shape)
    #     # prediction = self._read_out(H)
    #     # print("Encoder output shape: ", prediction.shape)
    #     flatten_output = torch.flatten(H, start_dim=1)
    #     # print(flatten_output.shape)
    #     prediction = self._read_out(flatten_output)
    #     # print(prediction.shape)
    #     if self.predict_cov:
    # #         prediction = prediction.view(-1, self.n_dims, self.n_dims)
    #     return prediction


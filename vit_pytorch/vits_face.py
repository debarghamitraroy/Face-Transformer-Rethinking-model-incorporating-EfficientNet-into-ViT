import math
import torch
from torch import nn
from einops import repeat
from einops import rearrange
from torch.nn import Parameter
import torch.nn.functional as F

MIN_NUM_PATCHES = 16


# ======= SoftMax Loss =======#
class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                   if device_id=None, it will be trained on CPU without model parallel.
    """

    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, label):
        if self.device_id is None:
            out = F.linear(input, self.weight, self.bias)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat(
                    (out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1
                )
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


# ======= ArcFace Loss =======#
class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                   if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta+m)
    """

    def __init__(
        self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s: float = s
        self.m: float = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin: bool = easy_margin
        self.cos_m: float = math.cos(self.m)
        self.sin_m: float = math.sin(self.m)
        self.th: float = math.cos(math.pi - m)
        self.mm: float = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======#
        if self.device_id is None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # ======= Convert label to one-hot =======#
        one_hot = torch.zeros(cosine.size())
        if self.device_id is not None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# ======= CosFace Loss =======#
class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s: float = s
        self.m: float = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======#
        if self.device_id is None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(
                self.weight,
                len(self.device_id),
                dim=0,
            )
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        phi = cosine - self.m

        # ======= convert label to one-hot =======#
        one_hot = torch.zeros(cosine.size())
        if self.device_id is not None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)

        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features = "
            + str(self.in_features)
            + ", out_features = "
            + str(self.out_features)
            + ", s = "
            + str(self.s)
            + ", m = "
            + str(self.m)
            + ")"
        )


# ======= SFace Loss =======#
class SFaceLoss(nn.Module):
    r"""Implement of SFace (https://arxiv.org/pdf/2205.12010.pdf):"""

    def __init__(
        self, in_features, out_features, device_id, s=64.0, k=80.0, a=0.90, b=1.2
    ):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s: float = s
        self.k: float = k
        self.a: float = a
        self.b: float = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======
        if self.device_id is None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])

                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        # ======= s*cos(theta) =======#
        output = cosine * self.s

        # ======= SFaceLoss =======#
        one_hot = torch.zeros(cosine.size())
        if self.device_id is not None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        zero_hot = torch.ones(cosine.size())
        if self.device_id is not None:
            zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, label.view(-1, 1), 0)

        WyiX = torch.sum(one_hot * output, 1)
        with torch.no_grad():
            theta_yi = torch.acos(WyiX / self.s)
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra_loss = -weight_yi * WyiX

        Wj = zero_hot * output
        with torch.no_grad():
            theta_j = torch.acos(Wj / self.s)
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter_loss = torch.sum(weight_j * Wj, 1)

        loss = intra_loss.mean() + inter_loss.mean()
        Wyi_s = WyiX / self.s
        Wj_s = Wj / self.s
        return (
            output,
            loss,
            intra_loss.mean(),
            inter_loss.mean(),
            Wyi_s.mean(),
            Wj_s.mean(),
        )


# ======= Residual =======#
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# ======= PreNorm =======#
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ======= Feed Forward =======#
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ======= Attention =======#
class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()

        self.heads: int = heads
        inner_dim = dim_head * heads
        self.scale: float = dim**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] & mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


# ======= Transformer =======#
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = Residual(
            PreNorm(
                dim,
                Attention(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                ),
            )
        )

        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# ======= ViTs_face =======#
class ViTs_face(nn.Module):
    def __init__(
        self,
        *,
        loss_type,
        GPU_ID,
        num_class,
        image_size,
        patch_size,
        ac_patch_size,
        pad,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * ac_patch_size**2
        assert num_patches > MIN_NUM_PATCHES, (
            f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        )
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_size = patch_size
        self.soft_split = nn.Unfold(
            kernel_size=(ac_patch_size, ac_patch_size),
            stride=(self.patch_size, self.patch_size),
            padding=(pad, pad),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool: str = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == "None":
            print("no loss for vit_face")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "SFaceLoss":
                self.loss = SFaceLoss(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )

    def forward(self, img, label=None, mask=None):
        x = self.soft_split(img).transpose(1, 2)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

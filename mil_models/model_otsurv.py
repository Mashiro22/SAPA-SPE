import math

import torch
from torch import nn

from .components import process_surv, replace_nan
from .otsurv_component.ot_attn import OT_Attn


class FixedRandomOrthogonalPatchEncoder(nn.Module):
    """Fixed random orthogonal projection for UNI patch features."""

    def __init__(self, patch_dim=1024, hidden_dim=256, seed=1):
        super().__init__()
        patch_dim = int(patch_dim)
        hidden_dim = int(hidden_dim)
        if hidden_dim > patch_dim:
            raise ValueError(
                f"random_orthogonal requires hidden_dim <= patch_dim; "
                f"got hidden_dim={hidden_dim}, patch_dim={patch_dim}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        basis = torch.randn(patch_dim, patch_dim, generator=generator)
        q, r = torch.linalg.qr(basis, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        scale = math.sqrt(float(patch_dim) / float(hidden_dim))
        w = q[:, :hidden_dim].T.contiguous() * scale
        self.register_buffer("W", w)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x):
        h = x @ self.W.to(dtype=x.dtype).T
        return self.norm(h)


class OTSurv(nn.Module):
    """
    OTSurv: A Novel Multiple Instance Learning Framework for Survival Prediction 
    with Heterogeneity-aware Optimal Transport.

    Args:
        num_classes (int): Number of output classes for survival prediction
        patch_dim (int): Input feature dimension of patches
        hidden_dim (int): Hidden dimension for feature encoding
        num_prototypes (int): Number of learnable prototype embeddings for OT aggregation
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
            self,
            num_classes=1,
            patch_dim=1024,
            hidden_dim=256,
            num_prototypes=16,
            dropout_rate=0.25,
            patch_encoder_type="original",
            patch_encoder_seed=1
            ):
        super().__init__()
        
        self.patch_encoder_type = str(patch_encoder_type)
        self.patch_encoder_seed = int(patch_encoder_seed)
        if self.patch_encoder_type == "original":
            self.patch_encoder = nn.Sequential(
                nn.Linear(patch_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        elif self.patch_encoder_type == "random_orthogonal":
            self.patch_encoder = FixedRandomOrthogonalPatchEncoder(
                patch_dim=patch_dim,
                hidden_dim=hidden_dim,
                seed=self.patch_encoder_seed,
            )
        else:
            raise ValueError(
                f"Unsupported patch_encoder_type={self.patch_encoder_type!r}; "
                "expected 'original' or 'random_orthogonal'."
            )
        
        # Optimal transport attention mechanism for heterogeneity-aware aggregation
        self.ot_attn = OT_Attn(impl="hot")
        
        # Learnable prototype embeddings for optimal transport
        self.prototype_embeddings = nn.Embedding(num_prototypes, hidden_dim)
        
        # Linear layer for prototype aggregation
        self.linear = nn.Linear(num_prototypes, 1)
        
        # Final classifier for survival prediction
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward_no_loss(self, wsi_features, return_attn=False):
        """
        Forward pass without computing loss.
        
        Args:
            wsi_features: List of WSI Features. The last two elements are:
                   - wsi_features[-2]: iterations_per_epoch
                   - wsi_features[-1]: iterations
            return_attn: Whether to return attention weights

        """
        *features_list, iterations_per_epoch, iterations = wsi_features

        h_path_list = []
        Attn_OT_list = []
        for features in features_list:
            # Encode patch features
            encoded_features = self.patch_encoder(features)
            encoded_features = replace_nan(encoded_features)

            # Apply optimal transport attention
            # encoded_features:[Ni, D]
            # self.prototype_embeddings.weight:[K, D]
            # Attn_OT:[Ni, K]
            Attn_OT, _ = self.ot_attn(encoded_features, self.prototype_embeddings.weight, iterations, iterations_per_epoch)
            aggregated_feature = torch.mm(Attn_OT.T, encoded_features)
            
            h_path_list.append(aggregated_feature.unsqueeze(0))
            Attn_OT_list.append(Attn_OT)

        # h_path: [B, K, D]
        h_path = torch.cat(h_path_list, dim=0)

        # h_path: [B, D]
        h_path = self.linear(h_path.transpose(-1, -2)).squeeze(-1)

        # logits: [B, 1]
        logits = self.classifier(h_path)

        out = {'logits': logits}
        if return_attn:
            out['Attn_OT'] = Attn_OT_list

        return out

    def forward(self, x_path, return_attn=False, label=None, censorship=None, loss_fn=None):
        out = self.forward_no_loss(x_path, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict


class OTSurvProtoGate(OTSurv):
    """OTSurv variant with slide-adaptive prototype attention and a global residual gate."""

    def __init__(self, num_classes=1, patch_dim=1024, hidden_dim=256,
                 num_prototypes=16, dropout_rate=0.25, gate_hidden_dim=128,
                 patch_encoder_type="original", patch_encoder_seed=1):
        super().__init__(
            num_classes, patch_dim, hidden_dim, num_prototypes, dropout_rate,
            patch_encoder_type=patch_encoder_type,
            patch_encoder_seed=patch_encoder_seed,
        )
        self.proto_attn = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(gate_hidden_dim, 1)
        )
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def _proto_attention_stats(proto_scores, proto_weights):
        eps = 1e-8
        num_proto = proto_weights.shape[-1]
        entropy = -(proto_weights * proto_weights.clamp_min(eps).log()).sum(dim=-1)
        denom = torch.log(torch.tensor(
            float(max(num_proto, 2)),
            device=proto_weights.device,
            dtype=proto_weights.dtype,
        ))
        norm_entropy = entropy / denom.clamp_min(eps)
        max_weight = proto_weights.max(dim=-1).values
        effective_num = torch.exp(entropy)
        score_std = proto_scores.std(dim=-1, unbiased=False).mean()
        weight_std = proto_weights.std(dim=-1, unbiased=False).mean()
        return norm_entropy.mean(), max_weight.mean(), effective_num.mean(), score_std, weight_std

    def forward_no_loss(self, wsi_features, return_attn=False):
        *features_list, iterations_per_epoch, iterations = wsi_features

        h_path_list = []
        Attn_OT_list = []
        proto_weight_list = []
        proto_entropy_list = []
        proto_max_weight_list = []
        proto_effective_num_list = []
        proto_score_std_list = []
        proto_weight_std_list = []
        for features in features_list:
            encoded_features = self.patch_encoder(features)
            encoded_features = replace_nan(encoded_features)

            Attn_OT, _ = self.ot_attn(
                encoded_features,
                self.prototype_embeddings.weight,
                iterations,
                iterations_per_epoch
            )
            proto_features = torch.mm(Attn_OT.T, encoded_features)
            proto_scores = self.proto_attn(proto_features).transpose(0, 1)
            proto_weights = torch.softmax(proto_scores, dim=-1)
            h_ot = torch.mm(proto_weights, proto_features).squeeze(0)

            h_global = encoded_features.mean(dim=0)
            gate = self.global_gate(torch.cat([h_ot, h_global], dim=-1))
            h_path = gate * h_ot + (1.0 - gate) * h_global

            h_path_list.append(h_path.unsqueeze(0))
            Attn_OT_list.append(Attn_OT)
            proto_weight_list.append(proto_weights)
            proto_entropy, proto_max_weight, proto_effective_num, proto_score_std, proto_weight_std = self._proto_attention_stats(
                proto_scores, proto_weights
            )
            proto_entropy_list.append(proto_entropy)
            proto_max_weight_list.append(proto_max_weight)
            proto_effective_num_list.append(proto_effective_num)
            proto_score_std_list.append(proto_score_std)
            proto_weight_std_list.append(proto_weight_std)

        h_path = torch.cat(h_path_list, dim=0)
        logits = self.classifier(h_path)

        out = {
            "logits": logits,
            "proto_attention_entropy": torch.stack(proto_entropy_list).mean().detach(),
            "proto_attention_max_weight": torch.stack(proto_max_weight_list).mean().detach(),
            "proto_attention_effective_num": torch.stack(proto_effective_num_list).mean().detach(),
            "proto_attention_score_std": torch.stack(proto_score_std_list).mean().detach(),
            "proto_attention_weight_std": torch.stack(proto_weight_std_list).mean().detach(),
        }
        if return_attn:
            out["Attn_OT"] = Attn_OT_list
            out["proto_weights"] = proto_weight_list

        return out


class OTSurvPGSpatialPantherConcat(OTSurvProtoGate):
    """PG with PANTHER-inspired prototype semantics + spatial footprint concat.

    This keeps OT prototype features intact and concatenates a learned embedding
    of prototype-level spatial footprint descriptors before PG scoring/readout.
    It does not use additive token refinement and does not change OT assignment.
    """

    use_coords = True

    def __init__(self, num_classes=1, patch_dim=1024, hidden_dim=256,
                 num_prototypes=16, dropout_rate=0.25, gate_hidden_dim=128,
                 spatial_embed_dim=16, detach_stats=True, use_extent=False,
                 spatial_grid_size=8, top_mass_ratio=0.70,
                 top_mass_max_nodes=1024, eps=1e-8,
                 patch_encoder_type="original", patch_encoder_seed=1):
        super(OTSurvPGSpatialPantherConcat, self).__init__(
            num_classes=num_classes,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            dropout_rate=dropout_rate,
            gate_hidden_dim=gate_hidden_dim,
            patch_encoder_type=patch_encoder_type,
            patch_encoder_seed=patch_encoder_seed,
        )
        self.spatial_embed_dim = int(spatial_embed_dim)
        self.detach_stats = bool(detach_stats)
        self.use_extent = bool(use_extent)
        self.spatial_grid_size = int(spatial_grid_size)
        self.top_mass_ratio = float(top_mass_ratio)
        self.top_mass_max_nodes = int(top_mass_max_nodes)
        self.spatial_eps = float(eps)
        self.spatial_stat_dim = 5 if self.use_extent else 4
        ext_dim = hidden_dim + self.spatial_embed_dim

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.spatial_stat_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.spatial_embed_dim),
        )
        self.proto_attn_ext = nn.Sequential(
            nn.Linear(ext_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.global_gate_ext = nn.Sequential(
            nn.Linear(ext_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, ext_dim),
            nn.Sigmoid(),
        )
        self.classifier_ext = nn.Linear(ext_dim, num_classes)

    def _normalize_coords(self, coords, ref_tensor):
        if coords is None or not torch.is_tensor(coords):
            raise ValueError("OTSurvPGSpatialPantherConcat requires coords tensor; got missing or non-tensor coords.")
        if coords.dim() != 2 or coords.size(-1) < 2:
            raise ValueError("OTSurvPGSpatialPantherConcat requires coords with shape [N, 2].")
        coords = coords[:, :2].to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        xy_min = coords.min(dim=0).values
        xy_max = coords.max(dim=0).values
        scale = (xy_max - xy_min).max().clamp_min(self.spatial_eps)
        coords = (coords - xy_min) / scale
        return replace_nan(coords.clamp(0.0, 1.0))

    def _normalize_spatial_stats(self, stats):
        mean = stats.mean(dim=0, keepdim=True)
        std = stats.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.spatial_eps)
        return replace_nan((stats - mean) / std)

    def _select_top_mass_indices(self, q, mass_ratio=None):
        eps = self.spatial_eps
        ratio = self.top_mass_ratio if mass_ratio is None else float(mass_ratio)
        ratio = min(max(ratio, eps), 1.0)
        q = replace_nan(q).clamp_min(0.0)
        if q.numel() == 0:
            return torch.empty(0, device=q.device, dtype=torch.long)
        total = q.sum()
        sorted_q, sorted_idx = torch.sort(q, descending=True)
        if total <= eps:
            return sorted_idx[:1]
        cdf = torch.cumsum(sorted_q, dim=0) / total.clamp_min(eps)
        hit = torch.nonzero(cdf >= ratio, as_tuple=False)
        m = int(hit[0].item()) + 1 if hit.numel() > 0 else q.numel()
        return sorted_idx[:max(1, m)]

    def _topmass_mean_knn_distance(self, coords_for_knn, chunk_size=512):
        n = coords_for_knn.size(0)
        if n <= 1:
            return coords_for_knn.new_tensor(0.0)
        mins = []
        chunk_size = max(1, int(chunk_size))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d = torch.cdist(coords_for_knn[start:end], coords_for_knn, p=2)
            local = torch.arange(end - start, device=coords_for_knn.device)
            d[local, start + local] = float('inf')
            mins.append(d.min(dim=1).values)
        return torch.cat(mins, dim=0).mean()

    def _largest_component_mass_ratio_grid(self, top_coords, top_q):
        if top_coords.size(0) == 0:
            return top_coords.new_tensor(0.0)
        eps = self.spatial_eps
        g = max(int(self.spatial_grid_size), 2)
        top_w = replace_nan(top_q).clamp_min(0.0)
        top_w = top_w / top_w.sum().clamp_min(eps)
        idx = torch.clamp((top_coords * g).long(), min=0, max=g - 1)
        flat = idx[:, 1] * g + idx[:, 0]
        grid_mass = torch.zeros(g * g, device=top_coords.device, dtype=top_coords.dtype)
        grid_mass.scatter_add_(0, flat, top_w)
        active = grid_mass > 0
        total_mass = grid_mass.sum().clamp_min(eps)
        if not bool(active.any().item()):
            return top_coords.new_tensor(0.0)
        best_mass = top_coords.new_tensor(0.0)
        remaining = active.clone()
        for seed in range(g * g):
            if not bool(remaining[seed].item()):
                continue
            comp = torch.zeros_like(remaining)
            comp[seed] = True
            for _ in range(g * g):
                mat = comp.view(g, g)
                neigh = mat.clone()
                neigh[1:, :] |= mat[:-1, :]
                neigh[:-1, :] |= mat[1:, :]
                neigh[:, 1:] |= mat[:, :-1]
                neigh[:, :-1] |= mat[:, 1:]
                new_comp = neigh.view(-1) & active
                if torch.equal(new_comp, comp):
                    break
                comp = new_comp
            comp_mass = grid_mass[comp].sum()
            best_mass = torch.maximum(best_mass, comp_mass)
            remaining = remaining & (~comp)
        return best_mass / total_mass

    def _compute_spatial_stats(self, Attn_OT, coords):
        eps = self.spatial_eps
        coords = self._normalize_coords(coords, Attn_OT)
        if coords.size(0) != Attn_OT.size(0):
            raise ValueError(
                "OTSurvPGSpatialPantherConcat coords/attention length mismatch: "
                f"coords={coords.size(0)}, Attn_OT={Attn_OT.size(0)}"
            )
        attn = replace_nan(Attn_OT).clamp_min(0.0)
        _, k_num = attn.shape
        total_mass = attn.sum().clamp_min(eps)
        stats = []
        max_nodes = max(1, int(self.top_mass_max_nodes))
        for k in range(k_num):
            q = attn[:, k]
            mass = q.sum().clamp_min(eps)
            w = q / mass
            center = (w[:, None] * coords).sum(dim=0)
            spread = (w * (coords - center).pow(2).sum(dim=-1)).sum().clamp_min(eps).sqrt()

            idx = self._select_top_mass_indices(q, self.top_mass_ratio)
            top_coords = coords[idx]
            top_q = q[idx]
            top_w = top_q / top_q.sum().clamp_min(eps)
            top_center = (top_w[:, None] * top_coords).sum(dim=0)
            compactness = (top_w * (top_coords - top_center).pow(2).sum(dim=-1)).sum().clamp_min(eps).sqrt()

            struct_idx = idx[:min(idx.numel(), max_nodes)]
            struct_coords = coords[struct_idx]
            struct_q = q[struct_idx]
            mean_knn = self._topmass_mean_knn_distance(struct_coords)
            comp_ratio = self._largest_component_mass_ratio_grid(struct_coords, struct_q)

            vals = [spread, compactness, mean_knn, comp_ratio]
            if self.use_extent:
                vals.append(mass / total_mass)
            stats.append(torch.stack(vals))
        return replace_nan(torch.stack(stats, dim=0))

    def _encode_spatial_instances(self, features, coords):
        """Encode patch instances while preserving their coordinate correspondence."""
        return replace_nan(self.patch_encoder(features)), coords

    def forward_no_loss(self, wsi_features, return_attn=False):
        *items, iterations_per_epoch, iterations = wsi_features

        h_path_list = []
        Attn_OT_list = []
        proto_weight_list = []
        spatial_stats_raw_list = []
        spatial_stats_norm_list = []
        spatial_emb_list = []
        proto_entropy_list = []
        proto_max_weight_list = []
        proto_effective_num_list = []
        proto_score_std_list = []
        proto_weight_std_list = []
        spatial_metric_lists = [[] for _ in range(self.spatial_stat_dim)]

        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise ValueError("OTSurvPGSpatialPantherConcat requires input item as (features, coords).")
            features, coords = item[0], item[1]
            encoded_features, encoded_coords = self._encode_spatial_instances(
                features, coords
            )

            Attn_OT, _ = self.ot_attn(
                encoded_features,
                self.prototype_embeddings.weight,
                iterations,
                iterations_per_epoch,
            )
            Attn_OT = replace_nan(Attn_OT)
            proto_features = replace_nan(torch.mm(Attn_OT.T, encoded_features))

            spatial_stats_raw = self._compute_spatial_stats(Attn_OT, encoded_coords)
            spatial_stats_norm = self._normalize_spatial_stats(spatial_stats_raw)
            if self.detach_stats:
                spatial_stats_for_proj = spatial_stats_norm.detach()
            else:
                spatial_stats_for_proj = spatial_stats_norm
            spatial_emb = replace_nan(self.spatial_proj(spatial_stats_for_proj))
            proto_ext = replace_nan(torch.cat([proto_features, spatial_emb], dim=-1))

            proto_scores = replace_nan(self.proto_attn_ext(proto_ext).transpose(0, 1))
            proto_weights = replace_nan(torch.softmax(proto_scores, dim=-1))
            h_ot = replace_nan(torch.mm(proto_weights, proto_ext).squeeze(0))
            h_global = replace_nan(proto_ext.mean(dim=0))
            gate = replace_nan(self.global_gate_ext(torch.cat([h_ot, h_global], dim=-1)))
            h_path = replace_nan(gate * h_ot + (1.0 - gate) * h_global)

            h_path_list.append(h_path.unsqueeze(0))
            Attn_OT_list.append(Attn_OT)
            proto_weight_list.append(proto_weights)
            spatial_stats_raw_list.append(spatial_stats_raw.detach())
            spatial_stats_norm_list.append(spatial_stats_norm.detach())
            spatial_emb_list.append(spatial_emb.detach())
            proto_entropy, proto_max_weight, proto_effective_num, proto_score_std, proto_weight_std = self._proto_attention_stats(
                proto_scores, proto_weights
            )
            proto_entropy_list.append(proto_entropy)
            proto_max_weight_list.append(proto_max_weight)
            proto_effective_num_list.append(proto_effective_num)
            proto_score_std_list.append(proto_score_std)
            proto_weight_std_list.append(proto_weight_std)
            for i in range(self.spatial_stat_dim):
                spatial_metric_lists[i].append(spatial_stats_raw[:, i].mean())

        h_path = torch.cat(h_path_list, dim=0)
        logits = self.classifier_ext(h_path)

        out = {
            "logits": logits,
            "proto_attention_entropy": torch.stack(proto_entropy_list).mean().detach(),
            "proto_attention_max_weight": torch.stack(proto_max_weight_list).mean().detach(),
            "proto_attention_effective_num": torch.stack(proto_effective_num_list).mean().detach(),
            "proto_attention_score_std": torch.stack(proto_score_std_list).mean().detach(),
            "proto_attention_weight_std": torch.stack(proto_weight_std_list).mean().detach(),
            "spatial_spread_mean": torch.stack(spatial_metric_lists[0]).mean().detach(),
            "top70mass_spatial_compactness_mean": torch.stack(spatial_metric_lists[1]).mean().detach(),
            "top70mass_mean_knn_distance_mean": torch.stack(spatial_metric_lists[2]).mean().detach(),
            "top70mass_largest_component_mass_ratio_mean": torch.stack(spatial_metric_lists[3]).mean().detach(),
        }
        if self.use_extent:
            out["spatial_extent_mean"] = torch.stack(spatial_metric_lists[4]).mean().detach()
        if return_attn:
            out["Attn_OT"] = Attn_OT_list
            out["proto_weights"] = proto_weight_list
            out["spatial_stats"] = spatial_stats_raw_list
            out["spatial_stats_raw"] = spatial_stats_raw_list
            out["spatial_stats_norm"] = spatial_stats_norm_list
            out["spatial_emb"] = spatial_emb_list
        return out


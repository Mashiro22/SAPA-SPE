from mil_models.model_otsurv import OTSurv, OTSurvProtoGate, OTSurvPGSpatialPantherConcat


def create_survival_model(args):
    """Build one of the models included in the public SAPA-SPE release."""
    if args.loss_fn == "nll":
        num_classes = args.n_label_bins
    elif args.loss_fn in ("cox", "capped_cox", "rank"):
        num_classes = 1
    else:
        raise ValueError(f"Unsupported loss_fn: {args.loss_fn}")

    common = dict(
        num_classes=num_classes,
        patch_dim=args.feat_dim,
        hidden_dim=256,
        num_prototypes=16,
        patch_encoder_type=getattr(args, "patch_encoder_type", "original"),
        patch_encoder_seed=getattr(args, "patch_encoder_seed", 1),
    )

    if args.model_type == "otsurv":
        return OTSurv(**common)
    if args.model_type == "otsurv_pg":
        return OTSurvProtoGate(**common)
    if args.model_type == "sapa_spe":
        return OTSurvPGSpatialPantherConcat(
            **common,
            spatial_embed_dim=getattr(args, "spatial_embed_dim", 16),
            detach_stats=bool(getattr(args, "spatial_detach_stats", 1)),
            use_extent=bool(getattr(args, "spatial_use_extent", 0)),
            spatial_grid_size=getattr(args, "spatial_grid_size", 8),
            top_mass_ratio=getattr(args, "top_mass_ratio", 0.70),
            top_mass_max_nodes=getattr(args, "top_mass_max_nodes", 1024),
        )

    raise NotImplementedError(
        f"Unknown model_type={args.model_type!r}. "
        "Available models: otsurv, otsurv_pg, sapa_spe."
    )

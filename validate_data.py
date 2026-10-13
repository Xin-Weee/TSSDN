import json
from utils.config import parse_args
from utils.dataloder import build_graph_dataset_bundle
from utils.reproducibility import validate_inputs


def main():
    args = parse_args()
    report = validate_inputs(args)
    bundle = build_graph_dataset_bundle(
        args.processed_dir, args.feature_root, args.aum_root, args.seed,
        years=args.years, node_feat_dim=args.node_feat_dim,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        split_file=args.split_file)
    report["splits"] = {
        s: {"count": len(getattr(bundle, s + "_names")), "positive": int(getattr(bundle, s + "_labels").sum())}
        for s in ("train", "val", "test")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

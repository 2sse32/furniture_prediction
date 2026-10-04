from __future__ import annotations

import argparse
import json

from furniture_prediction import load_model, predict, train_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Furniture Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the best regression model")
    train_parser.add_argument("--data", required=True, help="Path to furniture CSV data")
    train_parser.add_argument("--target", default="price", help="Target column name")
    train_parser.add_argument("--model-output", default="model.joblib", help="Saved model path")

    predict_parser = subparsers.add_parser("predict", help="Predict furniture prices")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument(
        "--input-json",
        required=True,
        help='JSON object or array with furniture features, e.g. {"material": "wood", "width": 60}',
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        model_name, metrics = train_from_csv(
            csv_path=args.data,
            target_column=args.target,
            model_path=args.model_output,
        )
        print(f"Selected model: {model_name}")
        print(
            f"Metrics: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}"
        )
        print(f"Saved model to: {args.model_output}")
        return

    model = load_model(args.model)
    parsed = json.loads(args.input_json)
    records = parsed if isinstance(parsed, list) else [parsed]
    predictions = predict(model, records)
    print(json.dumps({"predictions": predictions.tolist()}))


if __name__ == "__main__":
    main()

import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from random import Random


def baseline_transformation(
    sessions: Dict[str, List[int]],
    num_neg: int = 99,
    seed: int = 42,
    user_session_path: Optional[str] = None,
    testing_path: Optional[str] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    rng = Random(seed)

    user_session: Dict[str, List[int]] = {}
    all_items_set: set[int] = set()

    for uid, items in sessions.items():
        user_session[uid] = items
        all_items_set.update(items)

    all_items_arr = np.fromiter(all_items_set, dtype=np.int32)
    testing_data: Dict[str, List[int]] = {}

    for uid, items in tqdm(user_session.items(), desc="Creating Data...", unit="user"):
        pos_item = items[-1]
        positives = set(items)

        negs: List[int] = []
        while len(negs) < num_neg:
            cand = rng.sample(list(all_items_arr), k=num_neg * 3)
            negs.extend(x for x in cand if x not in positives)
            negs = negs[:num_neg]

        testing_data[uid] = negs + [pos_item]

    def _dump(obj: dict, out_path: Optional[str]):
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    _dump(user_session, user_session_path)
    _dump(testing_data, testing_path)

    return user_session, testing_data


def main():
    parser = argparse.ArgumentParser(description="Baseline Session Transformation CLI")

    parser.add_argument("--input", type=str, required=True, help="Path to session .pkl file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files (no extension)")
    parser.add_argument("--num_neg", type=int, default=99, help="Number of negative samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    with open(args.input, "rb") as f:
        session_data = pickle.load(f)

    user_session_path = f"{args.output_prefix}_user_session.pkl"
    testing_path = f"{args.output_prefix}_test_data.pkl"

    print(f"📥 Input: {args.input}")
    print(f"💾 Saving user session to: {user_session_path}")
    print(f"💾 Saving test data to: {testing_path}")

    baseline_transformation(
        sessions=session_data,
        num_neg=args.num_neg,
        seed=args.seed,
        user_session_path=user_session_path,
        testing_path=testing_path,
    )


if __name__ == "__main__":
    main()

# run_active_learning.py
import argparse
from active_learning.loop import ActiveLearningLoop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)

    ap.add_argument("--meta_path", type=str, default="data/dataset/meta.h5")
    ap.add_argument("--fields_path", type=str, default="data/dataset/fields.npz")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--simulations_dir", type=str, default="simulations/")  # <- uzupełnij jeśli HPC

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--Tx_min", type=float, default=10e-9)
    ap.add_argument("--Tx_max", type=float, default=100e-9)
    ap.add_argument("--Tz_min", type=float, default=10e-9)
    ap.add_argument("--Tz_max", type=float, default=100e-9)

    ap.add_argument("--grid_points", type=int, default=40)
    ap.add_argument("--k_new", type=int, default=20)
    ap.add_argument("--mc_samples", type=int, default=10)

    args = ap.parse_args()

    al = ActiveLearningLoop(
        meta_path=args.meta_path,
        fields_path=args.fields_path,
        results_dir=args.results_dir,
        simulations_dir=args.simulations_dir,

        Tx_range=(args.Tx_min, args.Tx_max),
        Tz_range=(args.Tz_min, args.Tz_max),
        grid_points=args.grid_points,
        k_new=args.k_new,

        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # jeśli w loop.py chcesz mieć mc_samples jako parametr – dopniemy (patrz niżej)
    al.run(iterations=args.iterations)

if __name__ == "__main__":
    main()

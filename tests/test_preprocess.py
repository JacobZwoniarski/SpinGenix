import os
import SpinGenix.activate_path 
from SpinGenix.postprocess.preprocess import preprocess_simulation
# any existing .zarr file path
TEST_ZARR = "simulations/vx4/Tx_1.004e-07/Tz_5.4731e-08.zarr"

if __name__ == "__main__":
    if not os.path.exists(TEST_ZARR):
        print("ERROR: test .zarr not found:", TEST_ZARR)
        exit()

    field, (Tx, Tz), meta = preprocess_simulation(TEST_ZARR)

    print("Field shape:", field.shape)      # Expect (200,200,3)
    print("Tx:", Tx, "Tz:", Tz)
    print("Metadata:", meta)
    print("SUCCESS: Preprocessing works.")
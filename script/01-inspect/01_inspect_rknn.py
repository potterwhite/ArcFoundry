from rknn.api import RKNN
from pathlib import Path
import argparse


def inspect_single_rknn(model_path: Path, verbose: bool) :

    rknn = RKNN(verbose=verbose)

    print("\n---------- rknn.config ----------")
    ret = rknn.config(
        target_platform="rv1126b"
    )
    print(f"ret={ret}")

    #print("\n---------- rknn.load_onnx ----------")
    #ret = rknn.load_onnx(ONNX_MODEL)
    #print(f"ret={ret}")

    print("\n---------- rknn.load_rknn ----------")
    ret = rknn.load_rknn(str(model_path))
    print(f"ret={ret}")

    print("\n---------- rknn.init_runtime ----------")
    ret = rknn.init_runtime(
            target="rv1126b",
            perf_debug=True,
            eval_mem=True,
    )
    print(f"ret={ret}")

    print(f"--> eval_perf of RKNN model: {model_path}")
    tmp_perf_result = rknn.eval_perf(is_print=False)
    print(tmp_perf_result)

    print(f"--> eval_memory of RKNN model: {model_path}")
    tmp_memory_result = rknn.eval_memory()
    print(tmp_memory_result)

    #print("\n---------- rknn.init_runtime ----------")
    #ret = rknn.accuracy_analysis(inputs=PIC_PATH)
    #print(f"ret={ret}")

    rknn.release()


def main():

    PY_SCRIPT_DIR = Path(__file__).resolve().parent
    # ONNX_MODEL = "../models/onnx/encoder-epoch-99-avg-1.onnx"
    #ONNX_MODEL = f"{PY_SCRIPT_DIR}/../models/onnx_fixed/encoder_fixed.onnx"
    #RKNN_MODEL = f"{PY_SCRIPT_DIR}/../models/rknn/encoder.rknn"
    # DECODER_RKNN_MODEL="../models/rknn/decoder.rknn"
    # JOINER_RKNN_MODEL="../models/rknn/joiner.rknn"

    #PIC_PATH = f"{PY_SCRIPT_DIR}/../pictures/pictures.png"

    parser = argparse.ArgumentParser(
        description="Inspect one or more rknn models",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "model_paths",
        metavar="MODEL_PATH",
        type=Path,
        nargs="+",
        help="Path(s) to the rknn model file(s) to inspect."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print eval_perf & eval_memory of specific rknn you give"
    )

    args = parser.parse_args()  

    for model_path in args.model_paths:
        inspect_single_rknn(model_path, args.verbose)

if __name__ == "__main__":
    main()

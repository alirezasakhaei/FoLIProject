import torch
from src.models.mlp import mlp_1x512, mlp_3x512
from src.models.small_alexnet import small_alexnet
from src.models.inception import inception, inception_no_bn

def test_model(name, model_fn, shapes=[(1, 3, 32, 32), (1, 3, 224, 224)]):
    print(f"Testing {name}...")
    for shape in shapes:
        try:
            # Re-instantiate model for each shape because MLPs/AlexNet depend on input shape in constructor
            # For Inception, it's robust, but constructor arg is safer to respect.
            model = model_fn(input_shape=shape[1:]) 
            x = torch.randn(*shape)
            out = model(x)
            print(f"  Input {shape} -> Output {out.shape}")
            if out.shape[1] != 10:
                print(f"  WARNING: Expected 10 classes, got {out.shape[1]}")
        except Exception as e:
            print(f"  FAILED for shape {shape}: {e}")

if __name__ == "__main__":
    test_model("Small Inception", inception)
    test_model("Small Inception No BN", inception_no_bn)
    test_model("Small AlexNet", small_alexnet)
    test_model("MLP 1x512", mlp_1x512)
    test_model("MLP 3x512", mlp_3x512)

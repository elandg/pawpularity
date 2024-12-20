import sagemaker.pytorch

def main():
    estimator = sagemaker.pytorch.PyTorch(
        source_dir="scripts",
        entry_point="train.py",
        image_uri=image_uri,
        distribution={"pytorchddp": {"enabled": True}},
    )
    estimator.fit()


if __name__ == "__main__":
    main()
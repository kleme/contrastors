from datasets import load_dataset
import yaml
import fire

def main(ds_spec):
    with open(ds_spec) as stream:
        spec = yaml.safe_load(stream)

        [[load_dataset(ds['path']) for ds in lang_spec["datasets"]] for lang_spec in spec.values()]
        
if __name__ == "__main__":
    fire.Fire(main)
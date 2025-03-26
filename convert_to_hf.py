from argparse import ArgumentParser

from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from contrastors.models.huggingface import NomicBertConfig, NomicBertForPreTraining, NomicVisionModel
from sentence_transformers.models import Transformer, Pooling, Normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--biencoder", action="store_true")
    parser.add_argument("--vision", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.biencoder:
        config = BiEncoderConfig.from_pretrained(args.ckpt_path)
        model = BiEncoder.from_pretrained(args.ckpt_path, config=config)
        model = model.trunk
        model.save_pretrained("tmp")
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/Qwen2.5-Coder-1.5B-Instruct")
        tokenizer.save_pretrained("tmp")
        transformer = Transformer("tmp")
        pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=False, pooling_mode_lasttoken=True)
        norm = Normalize()
        model = SentenceTransformer(modules=[transformer, pooling, norm])
    elif args.vision:
        NomicBertConfig.register_for_auto_class()
        NomicVisionModel.register_for_auto_class("AutoModel")
        config = DualEncoderConfig.from_pretrained(args.ckpt_path)
        model = DualEncoder.from_pretrained(args.ckpt_path, config=config)
        vision = model.vision
        hf_config = NomicBertConfig(**model.vision.trunk.config.to_dict())
        model = NomicVisionModel(hf_config)

        state_dict = vision.state_dict()
        state_dict = {k.replace("trunk.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        config = NomicBertConfig.from_pretrained(args.ckpt_path)
        model = NomicBertForPreTraining.from_pretrained(args.ckpt_path, config=config)

    model.push_to_hub(args.model_name, private=args.private)

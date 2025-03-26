import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

from contrastors.dataset.text_text_loader import StreamingShardDataset, collate_fn, get_local_dataloader
from contrastors.dataset.text_code_loader import get_code_datasets, get_iterable_code_dataloader
from contrastors.distributed import gather_with_grad
from contrastors.loss import clip_loss, grad_cache_loss, clip_loss_pointwise
from contrastors.models import BiEncoder, BiEncoderConfig, LogitScale, CrossEncoder, CrossEncoderConfig, SimpleJointBiEncoder, Seq2SeqCrossEncoder
from contrastors.eval.codesage.utils import NL2CodeDataset
from .base import BaseTrainer


class TextTextTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(TextTextTrainer, self).__init__(config, dtype)
        self.use_grad_cache = config.train_args.grad_cache
        self.matryoshka_dims = config.train_args.matryoshka_dims
        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                config.train_args.matryoshka_loss_weights
                if config.train_args.matryoshka_dims and config.train_args.matryoshka_loss_weights
                else [1] * len(config.train_args.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

    def get_model(self, config):
        config = config.model_args
        if config.is_cross_encoder:
            cfg_class = CrossEncoderConfig
            if config.seq2seq:
                mdl_class =  Seq2SeqCrossEncoder
            else:
                mdl_class = CrossEncoder
        else:
            cfg_class = BiEncoderConfig
            if config.is_joint_encoder:
                mdl_class = SimpleJointBiEncoder
            else:
                mdl_class = BiEncoder

        if config.checkpoint is None:
            config = cfg_class(
                model_name=config.model_name,
                pooling=config.pooling,
                logit_scale=config.logit_scale,
                use_fused_kernels=config.use_fused_kernels,
                nomic_encoder=config.nomic_encoder,
                trainable_logit_scale=config.trainable_logit_scale,
                hamming=config.hamming,
                pretrained=config.pretrained,
                gradient_checkpointing=config.gradient_checkpointing,
                projection_dim=config.projection_dim,
            )
            model = mdl_class(config)
        else:
            self.print(f"Loading model from {config.checkpoint}")
            model_config = cfg_class.from_pretrained(config.checkpoint)
            if config.projection_dim is not None:
                model_config.projection_dim = config.projection_dim
            if config.gradient_checkpointing:
                model_config.gradient_checkpointing = True
            model = mdl_class.from_pretrained(config.checkpoint, config=model_config)

        if self.distributed and not self.deepspeed:
            model = model.to("cuda")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.process_index],
            )

        scale = LogitScale(config)

        if self.distributed and not self.deepspeed:
            scale = scale.to("cuda")
            if sum(p.requires_grad for p in scale.parameters()) > 0:
                scale = torch.nn.parallel.DistributedDataParallel(
                    scale,
                    device_ids=[self.process_index],
                )

        return {"model": model, "logit_scale": scale}

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        model_args = config.model_args
        gradient_accumulation_steps = train_args.gradient_accumulation_steps
        if data_config.streaming and not data_config.text_code:
            train_dataset = StreamingShardDataset(
                data_config.input_shards,
                data_config.batch_size,
                self.tokenizer,
                seed=data_config.seed,
                add_eos=model_args.nomic_encoder != True,
                add_prefix=model_args.add_prefix,
                num_negatives=model_args.num_negatives,
                download_locally=data_config.download,
                process_one_shard=data_config.process_one_shard,
                weighted_sampling=data_config.weighted_sampling,
                verbose=data_config.verbose,
            )
            if train_args.checkpoint is not None:
                print(f"Loading dataloader state from {train_args.checkpoint}")
                train_dataset.load_state(train_args.checkpoint)

            train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)
            self.print(f"Len of train_dataloader: {len(train_dataset)}")
            # round down in case

            self.total_num_steps = int(len(train_dataset) // data_config.batch_size)
        else:
            # config defines global batch size
            if data_config.batch_size % self.num_processes != 0:
                raise ValueError(
                    f"Batch size {data_config.batch_size} must be divisible by accelerator.num_processes {self.num_processes}"
                )

            batch_size = int(data_config.batch_size / self.num_processes)
            if data_config.text_code:
                if not hasattr(self, "cached_datasets"):
                    self.cached_datasets = get_code_datasets(data_config.input_shards, 
                            seed=data_config.seed, 
                            filter_type = data_config.filter_type, 
                            k=  data_config.filter_k, 
                            streaming= data_config.streaming, 
                            from_hub= data_config.from_hub, 
                            filter_256 = data_config.filter_256, 
                            tokenizer = self.tokenizer, 
                            num_negatives=model_args.num_negatives,
                            abs_neg_threshold= data_config.mine_abs_threshold, 
                            neg_perc_margain= data_config.neg_perc_margain, 
                            neg_abs_margain= data_config.neg_abs_margain, 
                            neg_start_range= data_config.neg_start_range, 
                            neg_end_range= data_config.neg_end_range, 
                            pre_apply_neg_mining= data_config.pre_apply_neg_mining, 
                            neg_sample_type= data_config.mine_type, 
                            temperature_start= data_config.temperature_start,
                            temperature_end= data_config.temperature_end, 
                            temp_decay_type = data_config.temp_decay_type
                            )

                if train_args.checkpoint is not None and 'step_' in train_args.checkpoint and not hasattr(self, "alr_loaded"):
                    print(f"Loading dataloader state from {train_args.checkpoint}")
                    loader_state_file = f'{train_args.checkpoint}/saved_ds_state.json'
                    with open(loader_state_file, 'r') as file:
                        loaded_state = json.load(file)
                else:
                    loaded_state = None
                train_dataloader = get_iterable_code_dataloader(
                    *self.cached_datasets,
                    batch_size,
                    self.tokenizer,
                    seed=data_config.seed,
                    add_prefix=model_args.add_prefix,
                    num_workers=data_config.workers,
                    epoch=epoch,
                    streaming= data_config.streaming, 
                    is_cross_encoder= self.config.model_args.is_cross_encoder, 
                    add_eos_mask= data_config.add_eos_mask, 
                    seq2seq= self.config.model_args.seq2seq, 
                    yielded= loaded_state['yielded'] if loaded_state is not None else None, 
                    global_batch_size= int(data_config.batch_size)
                )
                if loaded_state is not None:
                    train_dataloader.load_state(loaded_state)
                    self.alr_loaded = True
                self.total_num_steps = int(
                    train_dataloader.get_length() // data_config.batch_size
                )
            else:
                train_dataloader = get_local_dataloader(
                    data_config.input_shards,
                    batch_size,
                    self.tokenizer,
                    seed=data_config.seed,
                    num_negatives=model_args.num_negatives,
                    add_prefix=model_args.add_prefix,
                    num_workers=data_config.workers,
                    epoch=0,
                )
                self.total_num_steps = int(
                    len(train_dataloader.dataset) // data_config.batch_size
                )

        return {"train": train_dataloader, "val": None, "test": None}

    def save_model(self, output_dir):
        super().save_model(output_dir)
        if self.global_rank == 0:
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def forward_step(self, model, inputs, logit_scale, **kwargs):
        model.train()
        if self.use_grad_cache:
            loss = self._grad_cache_forward_step(model, inputs, logit_scale, **kwargs)
        else:
            loss = self._forward_step(
                model,
                inputs,
                logit_scale,
                matryoshka_dims=self.matryoshka_dims,
                matroyshka_loss_weights=self.matryoshka_loss_weights,
                **kwargs,
            )

        return loss

    def backward(self, loss):
        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            # grad cache backprops in the loss function, becomes a noop
            if not self.use_grad_cache:
                loss.backward()

    def _grad_cache_forward_step(self, model, batch, logit_scale, **kwargs):
        # TODO: could pass this to grad cache loss and log?
        batch.pop("dataset_name")
        kwargs.pop("step")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        query_inputs = {k.replace("query_", ""): v for k, v in batch.items() if "query" in k}
        document_inputs = {k.replace("document_", ""): v for k, v in batch.items() if "document" in k}
        loss = grad_cache_loss(
            tower1=model,
            tower2=model,
            t1_inputs=query_inputs,
            t2_inputs=document_inputs,
            chunk_size=self.config.train_args.chunk_size,
            logit_scale=logit_scale,
            **kwargs,
        )
        return loss
    
    def _cross_encoder_forward_step(self, model, batch, **kwargs):
        if self.config.model_args.seq2seq:
            out = model(**{k : v.to(model.device) 
                for k, v in batch.items() 
                if isinstance(v, torch.Tensor)}) 
            
            # if dist.get_rank() == 0:
            #     import pdb;pdb.set_trace()
            # dist.barrier()
            return out['loss']
            # return model(
            # input_ids=batch["input_ids"].to(model.device),
            # attention_mask=batch["attention_mask"].to(model.device),
            # labels = batch['labels'].to(model.device)
            # )['loss']
        dataset_name = batch.pop("dataset_name")
        query_outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device)
        )
        logits = query_outputs["ranking_score"].to(model.device)
        if self.config.train_args.loss_fn == 'clip':
            logits = gather_with_grad(logits)
            return clip_loss_pointwise(logits, self.config.model_args.num_negatives)
        else:
            labels = batch['labels'].to(model.device).reshape(-1, 1)
            loss_fct = nn.BCEWithLogitsLoss()
            return loss_fct(logits, labels)
        

    def _forward_step(self, model, batch, logit_scale, matryoshka_dims=None, matroyshka_loss_weights=None, **kwargs):
        if self.config.model_args.is_cross_encoder:
            return self._cross_encoder_forward_step(model, batch, **kwargs)
        normalize = True if matryoshka_dims is None else False
        dataset_name = batch.pop("dataset_name")
        gen_kwargs = {'is_code': False} if self.config.model_args.is_joint_encoder else {}
        query_outputs = model(
            input_ids=batch["query_input_ids"].to(model.device),
            attention_mask=batch["query_attention_mask"].to(model.device),
            normalize=normalize, **gen_kwargs
        )
        gen_kwargs = {'is_code': True} if self.config.model_args.is_joint_encoder else {}
        document_outputs = model(
            input_ids=batch["document_input_ids"].to(model.device),
            attention_mask=batch["document_attention_mask"].to(model.device),
            normalize=normalize, **gen_kwargs
        )
        if "negative_input_ids" in batch:
            negative_outputs = model(
                input_ids=batch["negative_input_ids"].to(model.device),
                attention_mask=batch["negative_attention_mask"].to(model.device),
                normalize=normalize,
            )

        queries = query_outputs["embedding"]
        all_documents = gather_with_grad(document_outputs["embedding"])

        if matryoshka_dims:
            loss = 0.0
            for loss_weight, dim in zip(matroyshka_loss_weights, matryoshka_dims):
                reduced_q = F.normalize(queries[:, :dim], dim=-1)
                reduced_d = F.normalize(all_documents[:, :dim], dim=-1)

                name_with_dim = f"{dataset_name}_matryoshka_{dim}"

                dim_loss = clip_loss(
                    query=reduced_q,
                    document=reduced_d,
                    logit_scale=logit_scale,
                    tracker=self.tracker,
                    dataset=name_with_dim,
                    **kwargs,
                )

                loss += loss_weight * dim_loss
        else:
            loss = clip_loss(
                query=queries,
                document=all_documents,
                logit_scale=logit_scale,
                tracker=self.tracker,
                dataset=dataset_name,
                **kwargs,
            )

        return loss

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_args=train_args,
            total_num_steps=total_num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if train_args.clamp_logits:
            with torch.no_grad():
                self.model["scale"].module.logit_scale.clamp_(0, np.log(train_args.logit_max))

        return loss

    def eval_loop(self, model, dataloader, step, logit_scale):
        raise NotImplementedError("CLIP Trainer does not support evaluation")
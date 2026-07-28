"""Continual pretraining.

Next-token prediction over packed biomedical text. This is the stage the project cares about
first: CPT is where broad domain knowledge, diversity and long-tail coverage are supposed to
enter the model, and therefore where corpus selection should matter most.

Written against transformers 5.x, which renamed ``torch_dtype`` to ``dtype`` and replaced
``Trainer(tokenizer=...)`` with ``processing_class``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.data.packing import pack_to_dataset
from medsel.registry import get_loader
from medsel.schema import CorpusDoc
from medsel.stages.base import Stage, StageResult
from medsel.utils.device import pick_device, resolve_dtype, supports_bf16
from medsel.utils.seed import set_seed

__all__ = ["CPTStage"]


class CPTStage(Stage):
    """Continual pretraining over a :class:`CorpusDoc` source."""

    name: ClassVar[str] = "cpt"

    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        # Base models frequently ship without a pad token. Packing produces uniform blocks so
        # padding is rarely used, but the collator still requires one to exist.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def prepare(self, tokenizer: Any = None) -> Any:
        """Load, filter and pack the corpus into fixed-length blocks.

        Callable on its own to inspect a corpus without touching a model - which is the point of
        keeping data assembly separate from training.
        """
        cfg = self.config

        # Validate the source before building a tokenizer, so a misconfigured run fails in
        # milliseconds instead of after a model download.
        loader = get_loader(cfg.data.source, **cfg.data.loader)
        if loader.record_type is not CorpusDoc:
            raise ValueError(
                f"CPT needs a corpus source yielding CorpusDoc, but {cfg.data.source!r} yields "
                f"{loader.record_type.__name__}. Use 'pubmed' here; "
                f"QA sets belong to the SFT stage."
            )

        tokenizer = tokenizer or self.tokenizer()
        docs = loader.load(cfg.data.split, limit=cfg.data.limit)
        return pack_to_dataset(
            docs,
            tokenizer,
            block_size=cfg.train.block_size,
            drop_remainder=cfg.train.drop_remainder,
        )

    def model(self) -> Any:
        from transformers import AutoModelForCausalLM

        cfg = self.config.model
        kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(cfg.dtype),
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)

        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=cfg.lora_target_modules,
                    task_type="CAUSAL_LM",
                ),
            )
            model.print_trainable_parameters()

        return model

    def run(self) -> StageResult:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        cfg = self.config
        set_seed(cfg.train.seed)

        tokenizer = self.tokenizer()
        dataset = self.prepare(tokenizer)
        model = self.model()

        device = pick_device()
        use_bf16 = device == "cuda" and supports_bf16(device)

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            learning_rate=cfg.train.learning_rate,
            num_train_epochs=cfg.train.num_train_epochs,
            max_steps=cfg.train.max_steps,
            warmup_ratio=cfg.train.warmup_ratio,
            weight_decay=cfg.train.weight_decay,
            lr_scheduler_type=cfg.train.lr_scheduler_type,
            logging_steps=cfg.train.logging_steps,
            save_steps=cfg.train.save_steps,
            save_total_limit=cfg.train.save_total_limit,
            seed=cfg.train.seed,
            report_to=cfg.train.report_to,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
            # Mixed precision only where it is actually a win; fp16 on CPU is slow and unstable.
            bf16=use_bf16,
            fp16=device == "cuda" and not use_bf16,
            use_cpu=device == "cpu",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            processing_class=tokenizer,
        )

        train_output = trainer.train()

        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        metrics: dict[str, Any] = dict(train_output.metrics)
        metrics["n_blocks"] = len(dataset)
        metrics["block_size"] = cfg.train.block_size
        metrics["tokens_seen"] = len(dataset) * cfg.train.block_size
        metrics["device"] = device
        manifest_path = self.write_manifest(metrics)

        return StageResult(
            stage=self.name,
            output_dir=self.output_dir,
            metrics=metrics,
            manifest_path=manifest_path,
        )

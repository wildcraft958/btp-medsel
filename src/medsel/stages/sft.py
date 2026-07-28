"""Supervised fine-tuning.

Instruction-response training over the three QA loaders, with the loss masked to the response so
the model is not rewarded for reproducing the question. Where CPT installs broad domain knowledge,
this stage is credited with task format and instruction following, which is exactly why the
project argues the two stages want different data.

Prompts come from :mod:`medsel.prompts` rather than being written here. Sharing one renderer with
the evaluator is deliberate: if the SFT format and the evaluation format drifted apart, every
"does CPT help downstream" comparison would silently confound a data effect with a formatting
effect.

Written against TRL 1.x, which takes a prompt-completion dataset and masks the prompt span itself
via ``completion_only_loss``, and against transformers 5.x (``processing_class``, not
``tokenizer``).

One open design question from the project whiteboard, recorded here so it is not lost: whether to
train this stage with a *combined* objective, next-token loss on raw biomedical text alongside
instruction loss on QA pairs, rather than running CPT and SFT strictly in sequence. The claim to
test is that mixing retains domain knowledge that pure SFT erodes. Section 4.2 of
docs/literature_review.md covers the evidence. Not implemented: it needs a second data source in
the config, and the ablation should be designed before the plumbing.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.prompts.templates import TEMPLATE_VERSION, render_prompt
from medsel.registry import get_loader
from medsel.schema import QAExample
from medsel.stages.base import Stage, StageResult
from medsel.utils.device import pick_device, resolve_dtype, supports_bf16
from medsel.utils.seed import set_seed

__all__ = ["SFTStage", "render_completion"]


def render_completion(example: QAExample, include_rationale: bool = False) -> str:
    """Render the supervised target for one example.

    The leading space belongs to the completion rather than the prompt, matching
    :func:`medsel.prompts.templates.render_continuation`, so that tokenizers which attach leading
    whitespace to a word produce the same tokens at training and at evaluation time.
    """
    if example.answer_key is None:
        raise ValueError(f"{example.uid} has no answer_key; it cannot be an SFT target")

    answer = f" {example.answer_key}"
    if include_rationale and example.rationale:
        # Answer first, then the justification. Putting the letter first keeps the target aligned
        # with what the evaluator scores, so an SFT'd model is not penalised at eval time for
        # having learned to reason before answering.
        answer = f"{answer}\n\nExplanation: {example.rationale.strip()}"
    return answer


class SFTStage(Stage):
    """Supervised fine-tuning over a :class:`QAExample` source."""

    name: ClassVar[str] = "sft"

    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def examples(self) -> list[Any]:
        """Load the labelled QA examples this stage will train on.

        Validates the source before anything expensive happens, so a misconfigured run fails in
        milliseconds rather than after a model download.
        """
        cfg = self.config
        loader = get_loader(cfg.data.source, **cfg.data.loader)

        if loader.record_type is not QAExample:
            raise ValueError(
                f"SFT needs a QA source yielding QAExample, but {cfg.data.source!r} yields "
                f"{loader.record_type.__name__}. Use 'medqa', 'medmcqa' or 'pubmedqa' here; "
                f"raw corpora belong to the CPT stage."
            )

        if not loader.has_labels(cfg.data.split):
            raise ValueError(
                f"{cfg.data.source}:{cfg.data.split} withholds gold answers, so it cannot supply "
                f"SFT targets. MedMCQA's 'test' split is the usual mistake; use 'train'."
            )

        return list(loader.load(cfg.data.split, limit=cfg.data.limit))

    def prepare(self, tokenizer: Any = None) -> Any:
        """Render examples into a prompt-completion dataset. No model, no optimiser.

        ``tokenizer`` is accepted for interface symmetry with :class:`CPTStage` and is unused:
        TRL tokenises the rendered strings itself, so nothing here needs a vocabulary.
        """
        from datasets import Dataset

        cfg = self.config
        rows = []
        skipped_unlabeled = 0

        for example in self.examples():
            if example.answer_key is None:
                # Individual rows can lack a key even in a labelled split.
                skipped_unlabeled += 1
                continue
            rows.append(
                {
                    "prompt": render_prompt(example),
                    "completion": render_completion(example, cfg.train.include_rationale),
                }
            )

        if not rows:
            raise ValueError(
                f"{cfg.data.source}:{cfg.data.split} produced no usable SFT rows "
                f"({skipped_unlabeled} example(s) had no answer_key)."
            )

        dataset = Dataset.from_list(rows)
        # Carried so run() can report it without re-deriving, and so prepare() stays inspectable
        # on its own.
        self._skipped_unlabeled = skipped_unlabeled
        return dataset

    def model(self) -> Any:
        from transformers import AutoModelForCausalLM

        cfg = self.config.model
        kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(cfg.dtype),
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        return AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)

    def peft_config(self) -> Any:
        """LoRA config, or None for a full finetune.

        Built here rather than wrapping the model as CPT does, because TRL applies PEFT itself
        when handed a ``peft_config``.
        """
        cfg = self.config.model
        if not cfg.use_lora:
            return None

        from peft import LoraConfig

        return LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    def run(self) -> StageResult:
        from trl import SFTConfig, SFTTrainer

        cfg = self.config
        set_seed(cfg.train.seed)

        tokenizer = self.tokenizer()
        dataset = self.prepare(tokenizer)
        model = self.model()

        device = pick_device()
        use_bf16 = device == "cuda" and supports_bf16(device)

        args = SFTConfig(
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
            max_length=cfg.train.max_length,
            # The whole point of this stage: score the response, not the question.
            completion_only_loss=True,
            bf16=use_bf16,
            fp16=device == "cuda" and not use_bf16,
            use_cpu=device == "cpu",
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=self.peft_config(),
        )

        train_output = trainer.train()

        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        metrics: dict[str, Any] = dict(train_output.metrics)
        metrics["n_examples"] = len(dataset)
        metrics["skipped_unlabeled"] = getattr(self, "_skipped_unlabeled", 0)
        metrics["max_length"] = cfg.train.max_length
        metrics["include_rationale"] = cfg.train.include_rationale
        # Recorded because SFT and evaluation must share a template version for any downstream
        # comparison to mean anything.
        metrics["template_version"] = TEMPLATE_VERSION
        metrics["device"] = device
        manifest_path = self.write_manifest(metrics)

        return StageResult(
            stage=self.name,
            output_dir=self.output_dir,
            metrics=metrics,
            manifest_path=manifest_path,
        )

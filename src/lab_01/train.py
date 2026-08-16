"""A small LoRA SFT experiment for SST-2."""

from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

from lab_01.data import load_sst2_experiment
from lab_01.evaluate import sst2_prompt
from lab_01.models import format_messages, load_model


QWEN3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def lora_config() -> LoraConfig:
    """Return the conservative LoRA configuration for Qwen3-0.6B."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=QWEN3_LORA_TARGET_MODULES,
        bias="none",
    )


def format_sst2_training_example(tokenizer, model_name: str, example: dict) -> str:
    """Render one SST-2 user prompt plus its exact JSON assistant completion."""
    messages = [
        {"role": "user", "content": sst2_prompt(example["sentence"])},
        {"role": "assistant", "content": example["target"]},
    ]
    return format_messages(tokenizer, model_name, messages, add_generation_prompt=False)


def train_sst2_lora(
    model_name: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "outputs/checkpoints/qwen3-0.6b-sst2-lora",
    seed: int = 42,
) -> None:
    """Fine-tune Qwen3-0.6B adapters on exactly the 500 SST-2 train examples."""
    dataset = load_sst2_experiment(seed=seed)["train"]
    model, tokenizer = load_model(model_name)
    model.config.use_cache = False

    train_dataset = dataset.map(
        lambda example: {"text": format_sst2_training_example(tokenizer, model_name, example)},
        remove_columns=dataset.column_names,
    )
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        bf16=True,
        max_length=512,
        dataset_text_field="text",
        packing=False,
        save_strategy="epoch",
        save_total_limit=3,
        save_only_model=True,
        logging_steps=10,
        report_to="none",
        optim="paged_adamw_8bit",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    print("--- TRAINING METADATA ---")
    print(f"model_name: {model_name}")
    print(f"total_parameters: {total_parameters}")
    print(f"trainable_lora_parameters: {trainable_parameters}")
    print(f"trainable_percentage: {100 * trainable_parameters / total_parameters:.4f}")
    print(f"per_device_batch_size: {args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"effective_batch_size: {effective_batch_size}")
    print(f"gradient_checkpointing: {args.gradient_checkpointing}")
    print(f"training_examples: {len(train_dataset)}")
    print(f"epochs: {args.num_train_epochs}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"max_sequence_length: {args.max_length}")
    print(f"output_directory: {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)

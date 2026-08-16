"""Reserved entry points for future SFT, LoRA, and QLoRA experiments."""


def sft(*args, **kwargs) -> None:
    raise NotImplementedError("SFT support will be added with PEFT and TRL.")


def lora(*args, **kwargs) -> None:
    raise NotImplementedError("LoRA support will be added with PEFT and TRL.")


def qlora(*args, **kwargs) -> None:
    raise NotImplementedError("QLoRA support will be added with PEFT, TRL, and bitsandbytes.")

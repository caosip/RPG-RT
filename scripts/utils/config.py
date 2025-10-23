from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TrainMode(Enum):
    QLORA = 'qlora'
    LORA = 'lora'
    FULL = 'full'


class TrainArgPath(Enum):
    SFT_LORA_QLORA_BASE = 'sft_args'
    DPO_LORA_QLORA_BASE = 'dpo_args'

@dataclass
class CommonArgs:

    local_rank: int = field(default=1, metadata={"help": "local rank"})

    train_args_path: TrainArgPath = field(default='sft_args', metadata={"help": "[sft_args,dpo_args]"})
    max_len: int = field(default=1024, metadata={"help": "max len"})
    max_prompt_length: int = field(default=512, metadata={
        "help": "max prompt length"})
    train_data_path: Optional[str] = field(default='./', metadata={"help": "train data path"})
    model_name_or_path: str = field(default='./', metadata={"help": "model name or path"})

    train_mode: TrainMode = field(default='lora',
                                  metadata={"help": "[qlora, lora, full]"})
    use_dora: bool = field(default=False, metadata={"help": "use dora"})

    task_type: str = field(default="dpo_single",
                           metadata={"help": "[pretrain, sft, dpo_multi, dpo_single]"})

    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
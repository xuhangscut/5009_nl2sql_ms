import mindspore
from mindnlp.transformers import MiniCPMForCausalLM, AutoModelForCausalLM
from mindnlp.transformers import AutoTokenizer, CodeLlamaTokenizer
import mindspore.dataset as ds
from mindnlp.dataset import load_dataset
from mindnlp.peft import (
    get_peft_model,
    LoraConfig,
    # LoftQConfig, 暂无支持
    TaskType
)
from mindnlp.engine import TrainingArguments
from mindnlp.engine import Trainer
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
# mindspore.set_context(pynative_synchronize=True)
#load model
model_name = "AI-ModelScope/CodeLlama-7b-Instruct-hf"
# model_name = "OpenBMB/MiniCPM-2B-dpo-fp16"
tokenizer = CodeLlamaTokenizer.from_pretrained(model_name, mirror="modelscope")

# for MiniCPM
# ps. when use minicpm tokenizer, need to modify the chat_template which is in tokenizer_json
# tokenizer = AutoTokenizer.from_pretrained(model_name, mirror="modelscope")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'  # deal with overflow issues when training in half-precision

model = AutoModelForCausalLM.from_pretrained(model_name, mirror="modelscope")

# for MiniCPM
# model = MiniCPMForCausalLM.from_pretrained(model_name, mirror="modelscope")

dataset = load_dataset('csv', data_files={
    'train': './data/nl2table/train_dataset.csv',  # 训练数据路径， 例如'nl2sql/data/nl2table/train_dataset.csv'
    'validation': './data/nl2table/val_dataset.csv'  # 验证数据路径， 例如'nl2sql/data/nl2table/val_dataset.csv'
})
peft_output_dir = "./peft_models/nl2table/peft_model_ms/1"  # PEFT层的输出路径，例如'nl2sql/peft_models/nl2table/peft_model'


def process_dataset(dataset, tokenizer, batch_size=6, max_seq_len=1024, shuffle=False):
    def formatting_prompt_func(user_message, assistant_message):
        message = [
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': assistant_message}
        ]
        text = tokenizer.apply_chat_template(message, tokenize=False)

        return text
    def merge_and_pad(prompt):
        tokenized = tokenizer(text=prompt, 
                              padding='max_length', 
                              truncation='only_first', 
                              max_length=max_seq_len
                            )
        return tokenized['input_ids'], tokenized['input_ids']

    dataset = dataset.project(columns=['prompt1', 'outputs1'])
    dataset = dataset.map(formatting_prompt_func, ['prompt1', 'outputs1'], ['prompt'])
    dataset = dataset.map(merge_and_pad, ['prompt'], ['input_ids', 'labels'])
    # type_cast_op = transforms.TypeCast(mstype.float32)
    # dataset = dataset.map(operations=type_cast_op, input_columns=['input_ids'])
    # dataset = dataset.map(operations=type_cast_op, input_columns=['labels'])

    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(batch_size)

    return dataset

train_dataset = process_dataset(dataset['train'], tokenizer, batch_size=1, max_seq_len=2100)
eval_dataset = process_dataset(dataset['validation'], tokenizer, batch_size=1, max_seq_len=2100)
print(next(train_dataset.create_tuple_iterator()))
print(train_dataset.get_dataset_size())


# peft training
## Initialize lora parameters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=32,
    use_rslora=True,
    lora_dropout=0.1,
    target_cells=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"
    ]
)

model = get_peft_model(model, peft_config)
# print(model)
model.print_trainable_parameters()

num_train_epochs = 1
fp16 = True
overwrite_output_dir = True
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16  
gradient_checkpointing = True
evaluation_strategy = "steps"
learning_rate = 5e-5
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
save_steps = 50
logging_steps = 10
load_best_model_at_end= False
packing = False
save_total_limit=3
neftune_noise_alpha=5

training_arguments = TrainingArguments(
    output_dir=peft_output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy=evaluation_strategy, eval_steps=0.15,
    max_grad_norm = max_grad_norm,
    auto_find_batch_size = auto_find_batch_size,
    save_total_limit = save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    neftune_noise_alpha= neftune_noise_alpha
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments
)
trainer.train()

trainer.model.save_pretrained(peft_output_dir)
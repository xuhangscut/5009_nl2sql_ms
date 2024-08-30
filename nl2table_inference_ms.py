import pandas as pd
import json
import sqlite3
from mindnlp.transformers import MiniCPMForCausalLM, AutoModelForCausalLM
from mindnlp.transformers import AutoTokenizer, CodeLlamaTokenizer
from mindnlp.transformers import StoppingCriteria
from mindnlp.peft import (
    get_peft_model,
    LoraConfig,
    # LoftQConfig, 暂无支持
    TaskType,
    PeftModel
)
import re
import mindspore

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [6203]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> mindspore.Tensor:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        is_done = self.eos_sequence in last_ids
        return mindspore.Tensor(is_done)

model_name = 'AI-ModelScope/CodeLlama-7b-Instruct-hf'  # 模型名称
# model_path = 'nl2sql/models/deepseek'  # 原始模型所存放的文件夹， 例如 ‘nl2sql/hf_models/deepseek’
peft_layer_path = './5009_nl2sql/peft_models/nl2table/peft_model_ms/1'  # 在finetuning中， PEFT层的输出路径，例如'nl2sql/peft_models/nl2table/peft_model1'
test_data_prompt1 = './5009_nl2sql/data/nl2table/prompt1_test.json'  # 测试集prompt1的路径，例如'nl2sql/data/nl2table/prompt1_test.json'
test_data_output1 = './5009_nl2sql/data/nl2table/outputs1_test_ms.json'  # 测试集output1路径， 例如'nl2sql/data/nl2table/outputs1_test.json'


model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, peft_layer_path)
model = model.merge_and_unload()
print('model merge succeeded')
tokenizer = CodeLlamaTokenizer.from_pretrained(model_name, ms_dtype=mindspore.float16)
tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = tokenizer.bos_token_id
tokenizer.padding_side = 'left'
tokenizer.encode(' ;')

model.eval()

with open(test_data_prompt1, 'r') as f:
    prompt1 = json.load(f)

# This is for single sample inference
outputs1 = []
for i, prompt in enumerate(prompt1):
    print(f'{i} / {len(prompt1)}')
    message = [
        {'role': 'user', 'content': prompt.strip()}
    ]
    inputs = tokenizer.apply_chat_template(message, tokenize=True, return_tensors='ms', add_generation_prompt=True)
    responses = model.generate(
        inputs,
        max_new_tokens=50, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria = [EosListStoppingCriteria()]
    )
    output = tokenizer.decode(responses[0], skip_special_tokens=True).strip()
    outputs1.append(output)

    if (i + 1) % 10 == 0:
        with open(test_data_output1, 'w') as f:
            json.dump(outputs1, f)
        print(output)
with open(test_data_output1, 'w') as f:
            json.dump(outputs1, f)
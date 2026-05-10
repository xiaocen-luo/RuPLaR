import json
import torch
from torch.utils.data import Dataset
import re
import torch.nn.functional as F

def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data

class Stage2Dataset(Dataset):
    def __init__(
        self, 
        path,
        train_latent_soft_label_path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)
        
        self.train_latent_soft_label_path = train_latent_soft_label_path

        self.args = args
        self.model = model
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return pretrain_tokenize_function(
            examples = self.data[idx],
            latent_state=self._get_latent_state_manual_generation(self.data[idx], self.model),
            model = self.model,
            idx=idx
        )

    def extract_content_tokens_with_decimal(self, reasoning_step):
        """
        从推理步骤中提取内容token，支持小数点，忽略<<和>>
        输入: "<<20*1.20=24>>"
        输出: ["20", "*", "1.20", "=", "24"]
        """
        # 移除 << 和 >>
        clean_step = reasoning_step.strip('<>')

        # 支持小数的正则表达式
        # 匹配：数字（包括小数）、运算符
        decimal_pattern = r'(\d+\.?\d*|\+|\-|\*|\/|\=)'
        tokens = re.findall(decimal_pattern, clean_step)

        return tokens

    # step - 2
    def _get_latent_state_manual_generation_equal(self, examples, model):
        """返回手动生成标准的潜在状态（按等号分组，假设所有步骤都包含等号）"""
        if examples['cot'].startswith("<think>"):
            examples['cot'] = examples['cot'][len("<think>"):]

        if examples['cot'].endswith("</think>"):
            examples['cot'] = examples['cot'][:-len("</think>")]

        reasoning_steps = self.process_reasoning_chain(examples['cot'])
        latent_states = []

        for i, step in enumerate(reasoning_steps):
            content_tokens = self.extract_content_tokens_with_decimal(step)  # ["20", "*", "1.20", "=", "24"]

            # 查找等号的位置
            equal_index = -1
            for idx, token in enumerate(content_tokens):
                if token == "=":
                    equal_index = idx
                    break

            # 假设总是能找到等号
            # 第一组：等号之前的所有token
            before_equal = ''.join(content_tokens[:equal_index])
            probs_before = self.build_pair_temperature_based_target(before_equal, model)
            if probs_before.dim() == 1:
                probs_before = probs_before.unsqueeze(0)
            latent_states.append(probs_before)

            # 第二组：等号及等号之后的所有token
            after_equal = ''.join(content_tokens[equal_index:])
            probs_after = self.build_pair_temperature_based_target(after_equal, model)
            if probs_after.dim() == 1:
                probs_after = probs_after.unsqueeze(0)
            latent_states.append(probs_after)

        # 堆叠所有步骤
        if latent_states:
            latent_states_tensor = torch.cat(latent_states, dim=0)  # [num_steps, vocab_size]
            return latent_states_tensor

    # k - 2
    def _get_latent_state_manual_generation_k(self, examples, model, k=2):
        """返回手动生成标准的潜在状态"""
        if examples['cot'].startswith("<think>"):
            examples['cot'] = examples['cot'][len("<think>"):]

        if examples['cot'].endswith("</think>"):
            examples['cot'] = examples['cot'][:-len("</think>")]

        reasoning_steps = self.process_reasoning_chain(examples['cot'])

        latent_states = []
        for i, step in enumerate(reasoning_steps):
            content_tokens = self.extract_content_tokens_with_decimal(step)  # ["20", "*", "1.20", "=", "24"]
            # print("----------------------------content_tokens:---------------", content_tokens)
            # 为每两个内容token创建分布
            if len(content_tokens) >= k:
                # 按每两个token分组
                for j in range(0, len(content_tokens) - 1, k):
                    # token_pair = content_tokens[j:j + k]
                    token_pair = ''.join(content_tokens[j:j + k])
                    # print("token_pair:", token_pair)
                    probs = self.build_pair_temperature_based_target(token_pair, model)

                    # 确保 probs 是正确形状 [1, vocab_size]
                    if probs.dim() == 1:
                        probs = probs.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]

                    latent_states.append(probs)

                # 如果token数量为奇数，处理最后一个token
                if len(content_tokens) % k != 0:
                    # last_token = [content_tokens[-(len(content_tokens) % k):]]
                    last_token = ''.join(content_tokens[-(len(content_tokens) % k):])
                    # print("last_token:", last_token)
                    probs = self.build_pair_temperature_based_target(last_token, model)
                    if probs.dim() == 1:
                        probs = probs.unsqueeze(0)
                    latent_states.append(probs)
            else:
                # 如果内容token数量小于2，使用原始方法
                probs = self.build_temperature_based_target(step, model)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)
                latent_states.append(probs)

        # 堆叠所有步骤
        if latent_states:
            latent_states_tensor = torch.cat(latent_states, dim=0)  # [num_steps, vocab_size]
            # print(f"最终潜在状态形状: {latent_states_tensor.shape}")
            return latent_states_tensor

    def _get_latent_state_manual_generation(self, examples, model):
        """返回手动生成标准的潜在状态"""
        if examples['cot'].startswith("<think>"):
            examples['cot'] = examples['cot'][len("<think>"):]

        if examples['cot'].endswith("</think>"):
            examples['cot'] = examples['cot'][:-len("</think>")]

        reasoning_steps = self.process_reasoning_chain(examples['cot'])

        latent_states = []
        for i, step in enumerate(reasoning_steps):
            # probs = self.build_dirichlet_teacher_target(step, model) #bad
            # probs = self.build_gumbel_softmax_target(step, model)
            probs = self.build_mixture_target(step, model)

            # 确保 probs 是正确形状 [1, vocab_size]
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]

            # print(f"步骤 {i} 概率分布形状: {probs.shape}")
            latent_states.append(probs)

        # 堆叠所有步骤
        if latent_states:
            latent_states_tensor = torch.cat(latent_states, dim=0)  # [num_steps, vocab_size]
            # print(f"最终潜在状态形状: {latent_states_tensor.shape}")
            return latent_states_tensor

    # 方法一：使用温度softmax构建
    def build_pair_temperature_based_target(self, reasoning_step, model, temperature=1):
        """使用温度softmax构建分布"""
        tokenizer = model.tokenizer
        vocab_size = len(tokenizer)

        # 初始化为负无穷
        logits = torch.full((vocab_size,), -float('inf'), dtype=torch.bfloat16)

        key_tokens = self.extract_key_tokens(reasoning_step)
        # print("key_tokens:", key_tokens)

        # 给关键token设置logits
        for token in key_tokens:
            # 方法1：直接编码token字符串
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            # print(f"token '{token}' 编码为: {token_ids}")

            if token_ids:  # 如果编码成功
                for token_id in token_ids:
                    logits[token_id] = 2.0  # 基础分数
            else:
                print(f"警告: 无法编码token '{token}'")

        # 如果所有logits都是负无穷，需要处理
        if (logits == -float('inf')).all():
            print("警告: 所有logits都是负无穷，使用均匀分布作为后备")
            logits = torch.zeros(vocab_size, dtype=torch.bfloat16)

        # 应用温度softmax
        target_dist = F.softmax(logits / temperature, dim=-1)

        return target_dist

    # 方法一：使用温度softmax构建
    def build_temperature_based_target(self, reasoning_step, model, temperature=0.5):
        """
        使用温度softmax构建分布，确保不重要token概率为0
        """
        tokenizer = model.tokenizer
        vocab_size = len(model.tokenizer)

        # 关键修改：初始化为负无穷，softmax后概率为0
        logits = torch.full((vocab_size,), -float('inf'), dtype=torch.bfloat16)

        key_tokens = self.extract_key_tokens(reasoning_step)
        result_tokens = self.extract_result(reasoning_step)

        # 给关键token设置logits
        for token in key_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)
            if token_id != tokenizer.unk_token_id:
                logits[token_id] = 2.0  # 基础分数

        # 给结果token更高的logits
        for token in result_tokens:
            result_id = tokenizer.encode(token, add_special_tokens=False)
            if result_id != tokenizer.unk_token_id:
                logits[result_id] = 2.8  # 结果token分数更高

        # 应用温度softmax
        target_dist = F.softmax(logits / temperature, dim=-1)

        return target_dist

    # 方法一扩展：使用温度softmax构建 + Dirichlet噪声
    def build_dirichlet_teacher_target(self, reasoning_step, model,
                                       temperature=0.5,
                                       dirichlet_strength=0.1,
                                       use_expected=True):
        """
        用于KL散度训练的Dirichlet老师分布
        temperature: 控制确定性，越大越平滑
        dirichlet_strength: Dirichlet噪声强度，0=确定性温度softmax
        use_expected: 是否使用期望值（训练稳定） vs 采样（更多探索）
        """
        tokenizer = model.tokenizer
        vocab_size = len(tokenizer)

        # 先用温度softmax构建基础分布（保持原有逻辑）
        logits = torch.full((vocab_size,), -float('inf'), dtype=torch.bfloat16)

        key_tokens = self.extract_key_tokens(reasoning_step)
        result_token = self.extract_result(reasoning_step)

        # 给关键token设置logits
        for token in key_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)
            if token_id != tokenizer.unk_token_id:
                logits[token_id] = 2.0  # 基础分数

        # 给结果token更高的logits
        for token in result_token:
            result_id = tokenizer.encode(token, add_special_tokens=False)
            if result_id != tokenizer.unk_token_id:
                logits[result_id] = 2.8  # 结果token分数更高

        # 基础温度softmax分布
        base_dist = F.softmax(logits / temperature, dim=-1)

        if dirichlet_strength == 0:
            # 退化到原始温度softmax
            return base_dist

        # 转换为Dirichlet分布的alpha参数
        # alpha = base_dist * scale + epsilon (确保alpha>0)
        scale = dirichlet_strength * vocab_size
        epsilon = 1e-6  # 最小正值，确保数值稳定

        alpha = base_dist * scale + epsilon

        if use_expected:
            # 使用Dirichlet分布的期望值（训练更稳定）
            teacher_dist = alpha / alpha.sum()
        else:
            # 使用Dirichlet采样（提供更多变化）
            dirichlet_dist = torch.distributions.Dirichlet(alpha)
            teacher_dist = dirichlet_dist.sample()

        # 确保数值稳定
        teacher_dist = torch.clamp(teacher_dist, min=1e-8)
        teacher_dist = teacher_dist / teacher_dist.sum()  # 重新归一化

        return teacher_dist

    # 方法二：混合分布法
    def build_mixture_target(self, reasoning_step, model, result_weight=3.0):
        """
        构建混合分布：均匀分布 + 结果强调分布
        """
        tokenizer = model.tokenizer
        vocab_size = len(tokenizer)
        
            # 判断模型类型并设置 vocab_size
        model_name = model.config._name_or_path.lower()
        vocab_size = model.config.vocab_size
        #if 'deepseek' in model_name:
        #    vocab_size = model.config.vocab_size
        #    #vocab_size = len(tokenizer)
        #    print(f"[DEBUG] DeepSeek model detected, using config.vocab_size: {vocab_size}")
        #else:  # llama 或其他模型
        #    vocab_size = len(tokenizer)
        #    print(f"[DEBUG] Llama-like model detected, using tokenizer vocab_size: {vocab_size}")
    
        key_tokens = self.extract_key_tokens(reasoning_step)
        result_tokens = self.extract_result(reasoning_step)  # 改名为复数形式
        key_token_ids = []
        result_ids = []
        
        # 给关键token设置logits
        for token in key_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            # token_ids是一个列表，可能包含多个token id
            for token_id in token_ids:
                if token_id != tokenizer.unk_token_id:
                    key_token_ids.append(token_id)
    
        # 给结果token更高的logits
        for token in result_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            for token_id in token_ids:
                if token_id != tokenizer.unk_token_id:
                    result_ids.append(token_id)
    
        # 分布1：关键token均匀分布
        uniform_key = torch.zeros(vocab_size, dtype=torch.bfloat16)
        if key_token_ids:
            prob_per_key = 1.0 / len(key_token_ids)
            for token_id in key_token_ids:
                uniform_key[token_id] = prob_per_key
    
        # 分布2：结果token分布
        result_focus = torch.zeros(vocab_size, dtype=torch.bfloat16)
        if result_ids:  # 直接检查列表是否为空
            # 如果有多个结果token，可以均匀分配概率
            prob_per_result = 1.0 / len(result_ids)
            for token_id in result_ids:
                result_focus[token_id] = prob_per_result
    
        # 混合两个分布
        alpha = 0.2  # 均匀分布的权重
        target_dist = alpha * uniform_key + (1 - alpha) * result_focus
    
        # 确保归一化
        target_dist = target_dist / target_dist.sum()
    
        return target_dist

    def build_gumbel_softmax_target(self, reasoning_step, model, 
                                    temperature=0.5,
                                    hard=False,
                                    eps=1e-10):
        """
        使用Gumbel-Softmax构建老师分布
        hard=False: 使用soft relaxation（可微分）
        hard=True: 使用hard样本（近似one-hot）
        """
        tokenizer = model.tokenizer
        vocab_size = len(tokenizer)
        
        # 构建logits（与原函数相同）
        logits = torch.full((vocab_size,), -float('inf'), dtype=torch.bfloat16)
        
        key_tokens = self.extract_key_tokens(reasoning_step)
        result_token = self.extract_result(reasoning_step)
        key_score = self.get_learned_score(key_tokens, reasoning_step)
        result_score = self.get_learned_score(result_token, reasoning_step)
        
        # 给关键token设置logits
        for token in key_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)
            if token_id != tokenizer.unk_token_id:
                logits[token_id] = key_score  # 基础分数
        
        # 给结果token更高的logits
        result_id = tokenizer.encode(token, add_special_tokens=False)
        if result_id != tokenizer.unk_token_id:
            logits[result_id] = result_score  # 结果token分数更高
        
        # 应用Gumbel-Softmax
        # Gumbel-Softmax公式: y_i = exp((log(π_i) + g_i)/τ) / Σ_j exp((log(π_j) + g_j)/τ)
        # 其中g_i ~ Gumbel(0, 1)
        
        # 首先计算log-probabilities
        # 由于我们直接有logits，不需要先softmax再取log
        
        # 方法1：使用F.gumbel_softmax（PyTorch内置）
        target_dist = F.gumbel_softmax(logits, tau=temperature, hard=hard, eps=eps, dim=-1)
        
        return target_dist
        
    def get_learned_score(self, tokens, reasoning_step):
        step_tokens = reasoning_step.split()
        scores = []
        for token in tokens:
            if token in step_tokens:
                pos = step_tokens.index(token)
                pos_ratio = pos / len(step_tokens)  # 位置比例
                score = 1.5 + pos_ratio * 1.5  # 范围[1.5, 3.0]
                scores.append(score)
        return np.mean(scores) if scores else 2.0
    
    

    def process_reasoning_chain(self, full_cot):
        """
        处理完整的思维链，拆分成单个推理步骤
        输入: "<<18*12=216>> <<216-6*12=144>> <<144/12=12>>"
        输出: ["<<18*12=216>>", "<<216-6*12=144>>", "<<144/12=12>>"]
        """
        # 按 >> << 分割，但保留分隔符
        steps = re.findall(r'<<[^>]*>>', full_cot)
        return steps

    def extract_key_tokens(self, reasoning_step):
        """
        从推理步骤中提取关键token，小数点作为单独token
        输入: "<<20*1.20=24>>"
        输出: ["20", "*", "1", ".", "20", "=", "24"]
        """
        # 移除 << 和 >>
        clean_step = reasoning_step.strip('<>')

        # 小数点作为单独token
        tokens = re.findall(r'(\d+|\+|\-|\*|\/|\=|\.)', clean_step)

        return tokens

    def extract_result(self, reasoning_step):
        """
        从推理步骤中提取结果数字
        输入: "<<20/24=1.20>>"
        输出: ["1", ".", "20"]
        """
        clean_step = reasoning_step.strip('<>')

        if '=' in clean_step:
            result_part = clean_step.split('=')[-1].strip()
            # 匹配小数格式：整数部分、小数点、小数部分
            match = re.match(r'(\d+)(\.)(\d+)', result_part)
            if match:
                return [match.group(1), match.group(2), match.group(3)]

        # 如果没有匹配到小数格式，尝试提取数字
        numbers = re.findall(r'\d+', clean_step)
        if numbers:
            # 只返回最后一个数字（作为单个元素的列表）
            return [numbers[-1]]

        return []


def pretrain_tokenize_function(examples, 
        model, 
        latent_state,
        # sem_state,
        idx
    ):
     # Caution: Since each model uses a different instruction template, we apply custom formatting 
    # instead of using `apply_chat_template` directly.
    if 'deepseek' in model.model_path.lower():
        messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]

        if '</think>' in examples["cot_answer"] or '</think>' in examples["problem"]:
            raise ValueError("</think> triggers template logic — needs revision")
        input_text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
        input_prefix = input_text + "<｜Assistant｜>"
        input_suffix = examples["cot_answer"] + "<｜end▁of▁sentence｜>"
    elif 'llama' in model.model_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_suffix = examples["cot_answer"] + "<|eot_id|>"
    else:
        raise ValueError("Unsupported model type")

    input_prefix_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']
    input_suffix_ids = model.tokenizer(input_suffix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']

    input_text_ids = model.tokenizer(input_text, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']

    match = re.search(r'\\boxed\{\s*([^}]+?)\s*\}', examples["cot_answer"])
    ans_str = match.group(1).strip()
    ans_ids = model.tokenizer(ans_str).data["input_ids"]
    # print("ans_str", ans_str, "ans_ids", ans_ids)

    text_output = dict()

    num_latent_steps = latent_state.shape[0]
    # print("num_latent_steps", num_latent_steps)

    text_output['input_ids'] =input_prefix_ids + model.latent_token_ids[0] + [-100] * num_latent_steps+ model.latent_token_ids[1] + input_suffix_ids
    text_output['labels'] = [-100] * len(input_prefix_ids)+ [-100] * len(model.latent_token_ids[0]) + [-100] * num_latent_steps+ model.latent_token_ids[1] +input_suffix_ids
    latent_start_index = len(input_prefix_ids  + model.latent_token_ids[0])
    latent_end_index = latent_start_index + num_latent_steps

    text_output['latent_index'] = [latent_start_index, latent_end_index]
    text_output['latent_state']= latent_state
    # text_output['sem_state'] = sem_state
    text_output['q_and_step_index_list'] = [len(input_text_ids) - 1] + list(range(latent_start_index, latent_end_index + 1))
    # print("q_and_step_index_list", text_output['q_and_step_index_list'])
    # text_output['q_and_step_index_list'] = []
    # text_output['q_and_step_index_list'].append(len(input_text) - 1)
    # text_output['q_and_step_index_list'].append(latent_start_index)
    # text_output['q_and_step_index_list'].append(latent_end_index)
    text_output['ans_ids'] = ans_ids
        
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        latent_index = [example["latent_index"] for example in examples]
        latent_state = [example["latent_state"] for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        q_and_step_index_list = [torch.tensor(example["q_and_step_index_list"], dtype=torch.long) for example in examples]
        ans_ids = [torch.tensor(example["ans_ids"], dtype=torch.long) for example in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        
        labels = self.dynamic_padding(labels)

        batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "latent_state": latent_state,
            "latent_index": latent_index,
            "q_and_step_index_list": q_and_step_index_list,
            "ans_ids": ans_ids,
            "labels": labels}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
import os
import json
import random
import time
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 初始化客户端
client = OpenAI(
    api_key="sk-93817db303964020bbc79b017be4768b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0
)

# 意图配置
INTENTS = [
    {
        "name": "论文助手",
        "description": "生成学生日常学术问题",
        "seed_queries": [
            "论文的文献综述到底该咋写啊？",
            "结论部分一般都写些啥内容呢？",
            "学术论文格式有啥特别要求不？",
            "想找相关的文献，有啥好办法吗？",
            "研究假设怎么提才比较合理呢？"
        ],
        "sub_topics": ["研究设计", "文献分析", "方法创新", "结果解读", "学术规范", "投稿流程", "数据处理"],
        "speakers": ["本科生", "研究生", "刚接触科研的学生", "准备发表论文的人", "正在写毕业论文的学生"]
    },
    {
        "name": "心理助手",
        "description": "生成人们日常心理困扰问题",
        "seed_queries": [
            "最近老是焦虑，不知道该怎么办才好",
            "和朋友闹矛盾了，怎么处理比较好呢？",
            "压力太大了，有什么调节的好方法吗",
            "晚上总是睡不着，失眠好难受啊",
            "感觉自己自信心不够，怎么才能提升呢"
        ],
        "sub_topics": ["情绪管理", "人际关系", "自我认同", "压力应对", "成长困境", "沟通技巧", "心理健康"],
        "speakers": ["上班族", "学生", "刚毕业的年轻人", "家长", "老年人"]
    },
    {
        "name": "健身饮食助手",
        "description": "生成普通人健身饮食相关问题",
        "seed_queries": [
            "想增肌，该怎么制定计划才有效啊？",
            "减肥的时候到底该怎么吃啊？",
            "我是健身新手，适合做哪些运动呢？",
            "大家都说吃蛋白粉，有必要吗？",
            "有氧运动和无氧运动怎么搭配比较好"
        ],
        "sub_topics": ["增肌训练", "减脂饮食", "康复锻炼", "时间管理", "营养搭配", "运动计划", "食物营养成分"],
        "speakers": ["健身新手", "减肥人士", "办公室职员", "宝妈", "中年人群"]
    },
    {
        "name": "校园知识助手",
        "description": "生成学生校园生活相关问题",
        "seed_queries": [
            "奖学金怎么申请啊，有啥要求不？",
            "想转专业，需要满足什么条件呢？",
            "毕业论文要提前多久准备啊？",
            "在哪里能找到靠谱的实习机会呢",
            "纠结考研还是找工作，大家有啥建议吗"
        ],
        "sub_topics": ["升学规划", "科研实践", "校园生活", "就业准备", "政策解读", "社团活动", "学习方法", "竞赛加分"],
        "speakers": ["大一新生", "大二学生", "大三学生", "大四毕业生", "研究生"]
    }
]


def filter_low_quality_queries(queries, intent_name):
    """过滤低质量查询，但保留更多自然表达"""
    filtered = []
    for q in queries:
        if q and len(q.strip()) >= 4:
            clean_q = q.strip()
            # 允许包含数字，更宽松的过滤条件
            if any('\u4e00' <= c <= '\u9fff' for c in clean_q):
                filtered.append(clean_q)

    print(f"[{intent_name}] 过滤后保留 {len(filtered)}/{len(queries)} 条")
    return filtered


def generate_variant_queries(seed_queries, count):
    """生成更自然的变种查询，增加口语化表达"""
    variants = []
    # 更自然的表达方式转换
    filler_words = ["嗯，", "那个，", "想问一下，", "请教一下，", "我说，", "其实呢，"]
    hesitation_phrases = ["这个...", "就是...", "那个...", "呃..."]
    ending_phrases = ["呀", "呢", "哦", "啦", "啊", "的啦"]

    for _ in range(count):
        seed = random.choice(seed_queries)
        # 随机添加填充词
        if random.random() < 0.3:
            variant = random.choice(filler_words) + seed
        elif random.random() < 0.2:
            variant = random.choice(hesitation_phrases) + seed
        else:
            variant = seed

        # 随机修改结尾
        if random.random() < 0.25 and variant.endswith(('?', '？')):
            variant = variant[:-1] + random.choice(ending_phrases) + variant[-1]

        # 随机替换同义词
        synonym_replacements = {
            "怎么": ["如何", "怎样", "咋", "怎么个"],
            "如何": ["怎么", "怎样", "咋"],
            "吗": ["呢", ""],
            "呢": ["吗", ""],
            "我": ["我自己", "本人"],
            "想": ["想要", "想打算", "琢磨着"],
            "做": ["弄", "搞"]
        }

        if random.random() < 0.4:
            words = variant.split()
            new_words = []
            for word in words:
                for key, synonyms in synonym_replacements.items():
                    if key in word:
                        if random.random() < 0.3:
                            replacement = random.choice(synonyms)
                            word = word.replace(key, replacement)
                            break
                new_words.append(word)
            variant = ' '.join(new_words)

        if variant not in variants:
            variants.append(variant)
    return variants


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIError))
)
def generate_batch_queries(intent_config, batch_id, count):
    intent_name = intent_config["name"]
    try:
        # 随机选择说话人
        speaker = random.choice(intent_config["speakers"])
        # 确保有足够的种子问题
        sample_count = min(3, len(intent_config["seed_queries"]))
        current_seeds = random.sample(intent_config["seed_queries"], sample_count) if sample_count > 0 else \
        intent_config["seed_queries"][:1]

        current_sub_topic = random.choice(intent_config["sub_topics"])

        prompt = f"""请你模拟一个{speaker}的语气，围绕"{current_sub_topic}"这个主题，生成{count}个自然的问题或咨询。

参考以下问题的风格（但不要重复）：{', '.join(current_seeds)}

要求：
1. 用中文表达，就像平时说话一样自然
2. 可以包含口语化词汇、语气词
3. 可以是疑问句、陈述句或寻求建议的表达
4. 可以有重复意思但表达方式不同的句子
5. 不要太正式，避免学术化表达
6. 每个问题单独一行，不要编号和额外说明
7. 可以有一些不完美但真实的表达"""

        response = client.chat.completions.create(
            model="qwen-plus-2025-09-11",
            messages=[
                {"role": "system", "content": prompt.strip()},
                {"role": "user", "content": f"请生成{count}个符合要求的问题，越自然越好"}
            ],
            temperature=(0.8),  # 提高随机性
            max_tokens=count * 150  # 增加token限制
        )

        content = response.choices[0].message.content
        queries = [line.strip() for line in content.split('\n') if line.strip()]

        cleaned_queries = []
        for q in queries:
            # 移除可能的编号
            if q and q[0].isdigit() and (q[1] == '.' or q[1] == ')'):
                q = '.'.join(q.split('.')[1:]).strip()
            if q and len(q) > 3:
                cleaned_queries.append(q)

        print(f"[{intent_name}][批次{batch_id}] 生成 {len(cleaned_queries)} 条")
        return cleaned_queries[:count]

    except Exception as e:
        print(f"[{intent_name}][批次{batch_id}] 失败: {str(e)}")
        raise


def generate_intent_queries(intent_config, target=2000, max_workers=3):
    intent_name = intent_config["name"]
    total_queries = []
    save_file = f"{intent_name}_temp.json"
    output_file = f"{intent_name}_final_2000.json"

    # 加载中间结果
    if os.path.exists(save_file):
        try:
            with open(save_file, "r", encoding="utf-8") as f:
                total_queries = json.load(f)
            print(f"[{intent_name}] 已加载中间结果: {len(total_queries)} 条")
        except:
            total_queries = []

    # 确保种子问题在列表中
    seed_queries = intent_config["seed_queries"]
    for seed in seed_queries:
        if seed not in total_queries:
            total_queries.append(seed)

    print(f"[{intent_name}] 初始化后: {len(total_queries)} 条")

    start_time = time.time()
    batch_id = 0
    consecutive_failures = 0
    batch_size = 50

    while len(total_queries) < target and consecutive_failures < 50:
        remaining = target - len(total_queries)
        current_batch_size = min(batch_size, remaining)

        print(f"\n[{intent_name}] 当前进度: {len(total_queries)}/{target} | 批次大小: {current_batch_size}")

        try:
            batch_results = generate_batch_queries(intent_config, batch_id, current_batch_size)
            batch_id += 1

            if batch_results:
                consecutive_failures = 0  # 重置失败计数
                new_queries = filter_low_quality_queries(batch_results, intent_name)
                added_count = 0

                for q in new_queries:
                    # 放宽去重条件，允许一定相似性
                    similar_exists = any(
                        q in existing or existing in q
                        for existing in total_queries
                        if abs(len(q) - len(existing)) < 5
                    )

                    # 有20%概率允许相似问题
                    if (not similar_exists or random.random() < 0.2) and len(q) > 3:
                        total_queries.append(q)
                        added_count += 1
                        if len(total_queries) >= target:
                            break

                print(f"[{intent_name}] 新增: {added_count} 条 | 总计: {len(total_queries)} 条")

                # 定期保存
                if len(total_queries) % 50 == 0:
                    with open(save_file, "w", encoding="utf-8") as f:
                        json.dump(total_queries, f, ensure_ascii=False, indent=2)
                    print(f"[{intent_name}] 已保存中间结果")
            else:
                consecutive_failures += 1
                print(f"[{intent_name}] 连续失败: {consecutive_failures} 次")
        except Exception as e:
            consecutive_failures += 1
            print(f"[{intent_name}] 连续失败: {consecutive_failures} 次，错误: {str(e)}")

        # 当连续失败次数增加时，使用变种问题填充
        if consecutive_failures > 0 and consecutive_failures % 3 == 0:
            needed = min(20, target - len(total_queries))
            if needed > 0:
                variants = generate_variant_queries(seed_queries, needed)
                added = 0
                for variant in variants:
                    if len(total_queries) < target:
                        total_queries.append(variant)
                        added += 1
                print(f"[{intent_name}] 使用变种种子填充，新增 {added} 条，当前: {len(total_queries)} 条")

    # 最终处理
    final_queries = total_queries[:target]

    # 如果还不够，生成更多变种来补充
    if len(final_queries) < target:
        needed = target - len(final_queries)
        print(f"[{intent_name}] 生成额外 {needed} 条变种来满足目标数量")
        additional = generate_variant_queries(seed_queries, needed)
        final_queries.extend(additional[:needed])

    # 宽松去重，保留一些相似表达
    seen = set()
    unique_queries = []
    for q in final_queries:
        # 简单归一化处理
        normalized = q.lower().replace(' ', '').replace('？', '?')
        # 允许相似但不完全相同的表达
        if normalized not in seen:
            seen.add(normalized)
            unique_queries.append(q)

    # 如果仍然不够，再补充
    while len(unique_queries) < target:
        unique_queries.append(random.choice(unique_queries[:min(100, len(unique_queries))]))

    final_queries = unique_queries[:target]

    # 打乱顺序
    random.shuffle(final_queries)

    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_queries, f, ensure_ascii=False, indent=2)

    # 删除临时文件
    if os.path.exists(save_file):
        os.remove(save_file)

    print(f"\n[{intent_name}] 完成，耗时: {time.time() - start_time:.1f}s | 生成: {len(final_queries)} 条")
    return final_queries


if __name__ == "__main__":
    all_results = {}
    start_total = time.time()

    for intent in INTENTS:
        name = intent["name"]
        print(f"\n{'=' * 50}")
        print(f"开始生成 {name} 语料")
        print(f"{'=' * 50}")
        all_results[name] = generate_intent_queries(intent, target=2000)

    # 合并所有结果
    merged = []
    for intent_name, queries in all_results.items():
        merged.extend(queries)
        print(f"{intent_name}: {len(queries)} 条")

    # 打乱最终结果
    random.shuffle(merged)

    with open("all_intents_final_8000.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"全部完成")
    print(f"总耗时: {time.time() - start_total:.1f}s")
    print(f"总语料数量: {len(merged)}")
    print(f"{'=' * 50}")

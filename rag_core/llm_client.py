import json
import requests
from rag_core.config import RIGHT_CODE_API_KEY, RIGHT_CODE_BASE_URL, RIGHT_CODE_MODEL

def build_prompt(question, retrieved_chunks):
    context_text = "\n\n".join([format_context_chunk(idx, item) for idx, item in enumerate(retrieved_chunks)])
    return f"""
你是一个严谨的文档问答助手。请只根据下面的“参考资料”回答用户问题。

要求：
1. 只能使用参考资料中的信息，不要编造。
2. 如果资料中没有答案，请明确说“根据当前文档无法判断”。
3. 回答要清晰、分点，适合中文用户阅读。
4. 回答结尾必须列出你参考的资料来源，格式：文件名-页码-片段编号。

参考资料：
{context_text}

用户问题：
{question}

请给出答案：
""".strip()

def format_context_chunk(idx, item):
    page_text = f"第 {item['page']} 页" if item.get("page") else "无页码"
    return f"""
【资料片段 {idx + 1}】
文件名：{item["file_name"]}
页码：{page_text}
片段编号：{item["chunk_id"]}
相似度：{item["score"]:.4f}

内容：
{item["content"]}
""".strip()

def call_llm(question, retrieved_chunks):
    if not RIGHT_CODE_API_KEY:
        return "未检测到 RIGHT_CODE_API_KEY，请先在 .env 配置。"

    prompt = build_prompt(question, retrieved_chunks)
    headers = {"Content-Type":"application/json; charset=utf-8", "Authorization": f"Bearer {RIGHT_CODE_API_KEY}"}
    payload = {"model": RIGHT_CODE_MODEL, "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":prompt}]}], "stream": True}

    try:
        response = requests.post(RIGHT_CODE_BASE_URL, headers=headers, json=payload, stream=True, timeout=120)
    except Exception as exc:
        return f"请求模型 API 失败：{exc}"

    if response.status_code != 200:
        return f"模型 API 调用失败，状态码：{response.status_code}\n\n{response.text}"

    answer_parts = []
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line: continue
        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
        except: continue
        if not line.startswith("data:"): continue
        line = line[len("data:"):].strip()
        if line == "[DONE]": break
        try: data = json.loads(line)
        except: continue
        if data.get("type") == "response.output_text.delta":
            answer_parts.append(data.get("delta",""))
        choices = data.get("choices", [])
        for choice in choices:
            delta = choice.get("delta",{})
            if isinstance(delta, dict) and delta.get("content"):
                answer_parts.append(delta["content"])
            message = choice.get("message",{})
            if isinstance(message, dict) and message.get("content"):
                answer_parts.append(message["content"])
    final_answer = "".join(answer_parts).strip()
    if not final_answer:
        return "模型没有返回有效内容，请检查 API Key 或权限。"
    return final_answer
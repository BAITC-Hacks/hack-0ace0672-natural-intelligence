"""AI-ассистент аналитика. Владелец: narekdavtyan228-svg.

Модель deepseek-flash, OpenAI-совместимый /chat/completions, tool-calling.
Ключ ТОЛЬКО из переменной окружения DEEPSEEK_API_KEY — в код не вписываем.

Жёсткое правило: ассистент отвечает только числами из выгрузок и обязан
называть gid, на которые опирается. ТЗ запрещает чёрный ящик.
"""

import json
import os
import re
import urllib.request

from src import tools

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

SYSTEM = """Ты — ассистент AML-аналитика. Работаешь по графу внутрибанковских переводов.

ЖЕЛЕЗНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО ПО-РУССКИ.
2. Используй только числа, полученные из инструментов. Ничего не выдумывай.
3. Всегда называй конкретные gid, на которые опираешься.
4. Формулируй выводы как гипотезы для проверки, а не как утверждения о виновности.
5. Если данных не хватает — так и скажи и предложи, какой запрос сделать следующим.

Ограничения данных, о которых надо помнить: видны только исходящие переводы;
транзакции меньше 5000 KZT в выгрузку не попали; узлы 4-го колена обрезаны обходом,
их конечность не подтверждена; у seed-клиентов входящие суммы занижены."""

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_node", "description": "Карточка узла: роль, признаки, обоснование, топ-контрагенты",
        "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {
        "name": "who_collects_from", "description": "Кто собирает деньги с указанных узлов (получатели ниже по течению)",
        "parameters": {"type": "object", "properties": {
            "gids": {"type": "array", "items": {"type": "integer"}}}, "required": ["gids"]}}},
    {"type": "function", "function": {
        "name": "trace_path", "description": "Путь движения денег между двумя узлами",
        "parameters": {"type": "object", "properties": {
            "src": {"type": "integer"}, "dst": {"type": "integer"}}, "required": ["src", "dst"]}}},
    {"type": "function", "function": {
        "name": "top_nodes", "description": "Топ узлов по приоритету, можно фильтровать по роли и кластеру",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string"}, "cluster_id": {"type": "integer"}, "n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "cluster_summary", "description": "Сводка по кластеру: состав ролей, оборот, гипотеза",
        "parameters": {"type": "object", "properties": {"cluster_id": {"type": "integer"}}, "required": ["cluster_id"]}}},
    {"type": "function", "function": {
        "name": "find_nodes", "description": "Отбор узлов по признакам, например {'taint_share': '>0.8'}",
        "parameters": {"type": "object", "properties": {"filters": {"type": "object"}}, "required": ["filters"]}}},
]


def answer(question: str) -> dict:
    """Возвращает {"answer": str, "cited_gids": list[int], "mode": "llm"|"fallback"}."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return _fallback(question)
    try:
        return _ask_llm(question, key)
    except Exception as exc:  # сеть, лимит, что угодно — демо не должно падать
        res = _fallback(question)
        res["answer"] = f"[LLM недоступен: {exc}]\n\n" + res["answer"]
        return res


def _post(payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _ask_llm(question: str, key: str, max_rounds: int = 4) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_rounds):
        data = _post({"model": MODEL, "messages": messages, "tools": TOOLS_SPEC, "max_tokens": 1500}, key)
        msg = data["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            text = msg.get("content") or ""
            return {"answer": text, "cited_gids": _extract_gids(text), "mode": "llm"}
        for c in calls:
            fn = tools.TOOL_REGISTRY.get(c["function"]["name"])
            args = json.loads(c["function"]["arguments"] or "{}")
            result = fn(**args) if fn else {"error": "неизвестный инструмент"}
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
    return {"answer": "Не удалось завершить рассуждение за отведённое число шагов.",
            "cited_gids": [], "mode": "llm"}


def _extract_gids(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\b\d{15,20}\b", text)})


def _fallback(question: str) -> dict:
    """Шаблонные ответы на тех же функциях. Работает без сети и без ключа."""
    gids = [int(x) for x in re.findall(r"\b\d{15,20}\b", question)]
    if gids and len(gids) == 1:
        d = tools.get_node(gids[0])
        if "error" in d:
            return {"answer": d["error"], "cited_gids": [], "mode": "fallback"}
        txt = (f"Узел {d['gid']}: роль {d['role']} (уверенность {d['role_score']}), "
               f"кластер {d['cluster_id']}, приоритет {d['priority_score']}.\n{d['evidence']}")
        return {"answer": txt, "cited_gids": [d["gid"]], "mode": "fallback"}
    if gids:
        d = tools.who_collects_from(gids)
        lines = [f"  {c['gid']} — {c['role']}, достижим от {c['reached_from']} из указанных, "
                 f"приоритет {c['priority_score']}" for c in d["collectors"][:5]]
        return {"answer": "Собирают деньги с указанных узлов:\n" + "\n".join(lines),
                "cited_gids": [c["gid"] for c in d["collectors"][:5]], "mode": "fallback"}
    d = tools.top_nodes(n=5)
    lines = [f"  {x['gid']} — {x['role']}, приоритет {x['priority_score']}" for x in d["nodes"]]
    return {"answer": "Топ узлов по приоритету проверки:\n" + "\n".join(lines),
            "cited_gids": [x["gid"] for x in d["nodes"]], "mode": "fallback"}

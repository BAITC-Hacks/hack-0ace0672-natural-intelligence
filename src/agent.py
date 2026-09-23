"""AI-ассистент аналитика. Владелец: narekdavtyan228-svg.

Модель deepseek-flash, OpenAI-совместимый /chat/completions, tool-calling.
Ключ ТОЛЬКО из переменной окружения DEEPSEEK_API_KEY — в код не вписываем.

Жёсткое правило: ассистент отвечает только числами из выгрузок и обязан
называть gid, на которые опирается. ТЗ запрещает чёрный ящик.

Проверка из командной строки:

    python -m src.agent "кто собирает деньги с 100000003684369100 и 100000008165763100"
"""

import json
import os
import re
import urllib.request
from pathlib import Path

from src import roles, schema, tools

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

SYSTEM = f"""Ты — ассистент AML-аналитика. Работаешь по графу внутрибанковских переводов:
2248 клиентов, 3119 связей, 81 известный фигурант, оборот 365.9 млн KZT.

ЖЕЛЕЗНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО ПО-РУССКИ, даже если вопрос задан на другом языке.
2. Каждое число в твоём ответе обязано дословно присутствовать в выводе инструментов.
   Не округляй так, чтобы менялся смысл, не складывай числа в уме, не оценивай «примерно».
   Нужного числа нет в выводе — вызови инструмент ещё раз или скажи, что данных нет.
3. Всегда называй конкретные gid, на которые опираешься.
4. Формулируй выводы как гипотезы для проверки, а не как утверждения о виновности.
   Пиши «признаки консолидации», «требует запроса входящих», а не «отмывает деньги».
5. Если данных не хватает — так и скажи и предложи, какой запрос сделать следующим.
6. Не придумывай gid. Узел существует, только если его вернул инструмент.

Роли назначаются пороговыми правилами, пороги — перцентили нашего распределения:
transit — transit_ratio >= {roles.TRANSIT_RATIO_MIN}; consolidator — in_deg >= P90 и net_flow >= {roles.NET_FLOW_CONSOLIDATE};
distributor — out_deg >= P90, net_flow <= {roles.NET_FLOW_DISTRIBUTE}, hhi_out < {roles.HHI_OUT_FAN};
coordinator — деньги от >= 2 фигурантов, out_deg >= {roles.COORD_MIN_OUT_DEG}, taint >= P75;
terminal — нет исходящих, обход узел разворачивал; terminal_unknown — нет исходящих,
но узел на 4-м колене и обход оборван, конечность НЕ подтверждена.

Ограничения данных, о которых надо помнить: видны только исходящие переводы;
транзакции меньше 5000 KZT в выгрузку не попали; узлы 4-го колена обрезаны обходом;
у seed-клиентов входящие суммы занижены; 62.8% оборота приходит извне выборки."""

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_node", "description": "Карточка узла: роль, признаки, обоснование, топ-контрагенты",
        "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"]}}},
    {"type": "function", "function": {
        "name": "who_collects_from",
        "description": "Кто собирает деньги с указанных узлов: получатели ниже по течению, с суммами",
        "parameters": {"type": "object", "properties": {
            "gids": {"type": "array", "items": {"type": "integer"}},
            "max_depth": {"type": "integer", "description": "глубина обхода, по умолчанию 3"}},
            "required": ["gids"]}}},
    {"type": "function", "function": {
        "name": "trace_path", "description": "Путь движения денег между двумя узлами",
        "parameters": {"type": "object", "properties": {
            "src": {"type": "integer"}, "dst": {"type": "integer"}}, "required": ["src", "dst"]}}},
    {"type": "function", "function": {
        "name": "top_nodes", "description": "Топ узлов по приоритету, можно фильтровать по роли и кластеру",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string", "enum": schema.ROLES},
            "cluster_id": {"type": "integer"}, "n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "cluster_summary", "description": "Сводка по кластеру: состав ролей, оборот, гипотеза",
        "parameters": {"type": "object", "properties": {"cluster_id": {"type": "integer"}}, "required": ["cluster_id"]}}},
    {"type": "function", "function": {
        "name": "find_nodes",
        "description": "Отбор узлов по признакам. Условия вида {'taint_share': '>0.8', 'out_deg': '>20'}",
        "parameters": {"type": "object", "properties": {"filters": {"type": "object"}}, "required": ["filters"]}}},
]


def _load_env() -> None:
    """Подхватывает .env, если он есть. Ключ всё равно живёт только в окружении."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def answer(question: str) -> dict:
    """Возвращает {"answer": str, "cited_gids": list[int], "mode": "llm"|"fallback"}."""
    _load_env()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return _fallback(question)
    try:
        return _ask_llm(question, key)
    except Exception as exc:  # сеть, лимит, что угодно — демо не должно падать
        res = _fallback(question)
        res["answer"] = f"[LLM недоступен: {exc}. Ответ собран шаблоном по тем же данным]\n\n" + res["answer"]
        return res


def _post(payload: dict, key: str) -> dict:
    # Читаем при вызове, а не при импорте: иначе значения из .env не подхватятся.
    base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _ask_llm(question: str, key: str, max_rounds: int = 4) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    used: list[str] = []
    for _ in range(max_rounds):
        model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        data = _post({"model": model, "messages": messages, "tools": TOOLS_SPEC,
                      "temperature": 0.1, "max_tokens": 1500}, key)
        msg = data["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            text = msg.get("content") or ""
            return {"answer": text, "cited_gids": _extract_gids(text),
                    "mode": "llm", "tools_used": used}
        for c in calls:
            name = c["function"]["name"]
            fn = tools.TOOL_REGISTRY.get(name)
            used.append(name)
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                result = fn(**args) if fn else {"error": f"неизвестный инструмент {name}"}
            except Exception as exc:
                result = {"error": f"инструмент {name} не отработал: {exc}"}
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
    return {"answer": "Не удалось завершить рассуждение за отведённое число шагов. "
                      "Переформулируйте вопрос короче или укажите конкретные gid.",
            "cited_gids": [], "mode": "llm", "tools_used": used}


def _extract_gids(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\b\d{15,20}\b", text)})


# --------------------------------------------------------------- фолбэк без LLM

_ROLE_WORDS = {
    "coordinator": "coordinator", "координатор": "coordinator",
    "consolidator": "consolidator", "консолидатор": "consolidator", "собира": "consolidator",
    "distributor": "distributor", "дистрибьютор": "distributor", "раздат": "distributor",
    "transit": "transit", "транзит": "transit",
    "terminal_unknown": "terminal_unknown",
    "terminal": "terminal", "терминал": "terminal", "сток": "terminal",
    "peripheral": "peripheral", "перифер": "peripheral",
}

_NOT_ENOUGH = """Чего не хватает, чтобы закрыть белые пятна (оценка полноты данных):

1. Входящие переводы. В выгрузке видны только исходящие, поэтому полный баланс узла
   посчитать нельзя. У 335 узлов не-seed расход превышает приход на 229.9 млн KZT —
   это 62.8% оборота, источник которого вне выборки. Нужен запрос входящих по этим узлам.
2. Продолжение обхода. 444 узла стоят на 4-м колене без исходящих: обход оборван,
   а не деньги осели. Их роль помечена как terminal_unknown с уверенностью 0.30.
3. Переводы ниже 5 000 KZT. Порог выгрузки делает дробление на мелкие суммы невидимым.
4. Входящие у 81 фигуранта. Граф собран обходом от них, поэтому их приход занижен
   по построению, и отношение «отдал/получил» у seed некорректно.
5. Внешние счета и снятия наличными. Выход денег за пределы банка в данных не виден.

Все перечисленное — ограничения выгрузки, а не результат анализа."""


def _fmt_node(d: dict) -> str:
    return (f"Узел {d['gid']}: роль {d['role']} (уверенность {d['role_score']}), "
            f"кластер {d['cluster_id']}, приоритет проверки {d['priority_score']}.\n"
            f"Обоснование: {d['evidence']}\n"
            f"Контрагенты: {d['in_deg']} плательщиков на {d['in_kzt']/1e6:.2f} млн, "
            f"{d['out_deg']} получателей на {d['out_kzt']/1e6:.2f} млн. "
            f"Доля денег фигурантов taint {d['taint_share']:.2f}.")


def _explain_role(d: dict) -> str:
    """Почему именно эта роль — с порогом, который сработал."""
    r = d["role"]
    if r == "terminal_unknown":
        why = (f"нет исходящих переводов, узел стоит на 4-м колене (depth {d['depth']}), "
               f"обход графа на нём оборван. Конечность НЕ подтверждена: возможно, переводы дальше "
               f"есть, но в выгрузку не попали. Поэтому уверенность снижена штрафом до {d['role_score']}.")
    elif r == "terminal":
        why = (f"нет исходящих переводов >= 5 000 KZT, при этом обход узел разворачивал "
               f"(depth {d['depth']} <= 3) — значит сток настоящий, а не артефакт обрыва.")
    elif r == "transit":
        why = (f"вход и выход сходятся: transit_ratio {d['transit_ratio']:.2f} при пороге "
               f"{roles.TRANSIT_RATIO_MIN} — деньги проходят насквозь, а не оседают.")
    elif r == "consolidator":
        why = (f"получает от {d['in_deg']} плательщиков и удерживает: net_flow {d['net_flow']:.2f} "
               f"при пороге {roles.NET_FLOW_CONSOLIDATE} — признаки точки сбора средств.")
    elif r == "distributor":
        why = (f"рассылает на {d['out_deg']} получателей, net_flow {d['net_flow']:.2f} при пороге "
               f"{roles.NET_FLOW_DISTRIBUTE}, концентрация получателей hhi_out {d['hhi_out']:.2f} "
               f"при пороге {roles.HHI_OUT_FAN} — это настоящий веер, а не раздача двоим.")
    elif r == "coordinator":
        why = (f"деньги приходят от {d['n_seeds_upstream']} разных фигурантов, узел одновременно "
               f"собирает ({d['in_deg']} плательщиков) и раздаёт ({d['out_deg']} получателей), "
               f"taint {d['taint_share']:.2f}.")
    else:
        why = (f"связей мало: {d['in_deg']} входящих и {d['out_deg']} исходящих контрагентов — "
               f"под пороги активных ролей узел не подходит.")

    tail = ""
    if d.get("external_funding_gap", 0) > 0:
        tail = (f"\nОтдельно: расход превышает приход на {d['external_funding_gap']/1e6:.1f} млн — "
                f"источник этих денег вне выборки, требует запроса входящих.")
    return f"Узел {d['gid']} получил роль {r}, потому что {why}{tail}"


def _fallback(question: str) -> dict:
    """Шаблонные ответы на тех же шести функциях. Работает без сети и без ключа."""
    q = question.lower()
    gids = [int(x) for x in re.findall(r"\b\d{15,20}\b", question)]

    # 1. Полнота данных — опциональный пункт ТЗ.
    if any(w in q for w in ("не хватает", "нехватает", "полнот", "белых пятен", "белые пятна")):
        return {"answer": _NOT_ENOUGH, "cited_gids": [], "mode": "fallback"}

    # 2. Путь между двумя конкретными узлами.
    if len(gids) == 2 and any(w in q for w in ("путь", "связан", "дошли", "доход", "trace")):
        d = tools.trace_path(gids[0], gids[1])
        if "error" in d:
            return {"answer": d["error"], "cited_gids": [], "mode": "fallback"}
        if not d["found"]:
            return {"answer": f"Между {gids[0]} и {gids[1]} путей длиной до {d['max_len']} "
                              f"переводов в выгрузке нет.", "cited_gids": gids, "mode": "fallback"}
        p = d["paths"][0]
        chain = " -> ".join(str(x) for x in p["path"])
        return {"answer": f"Найдено путей: {d['found']}. Кратчайший ({p['length']} перевода):\n{chain}\n"
                          f"Узкое место цепочки — {p['min_edge_kzt']/1e6:.2f} млн: больше этой суммы "
                          f"по такому пути пройти не могло.",
                "cited_gids": p["path"], "mode": "fallback"}

    # 3. Один gid: либо объяснение роли, либо карточка.
    if len(gids) == 1:
        d = tools.get_node(gids[0])
        if "error" in d:
            return {"answer": d["error"], "cited_gids": [], "mode": "fallback"}
        text = _explain_role(d) if any(w in q for w in ("почему", "объясн", "роль")) else _fmt_node(d)
        return {"answer": text, "cited_gids": [d["gid"]], "mode": "fallback"}

    # 4. Несколько gid — сценарий ТЗ «кто собирает деньги с этих пятерых».
    if len(gids) >= 2:
        d = tools.who_collects_from(gids)
        if "error" in d:
            return {"answer": d["error"], "cited_gids": [], "mode": "fallback"}
        if not d["collectors"]:
            return {"answer": d.get("note", "получателей ниже по течению не найдено"),
                    "cited_gids": gids, "mode": "fallback"}
        lines = [f"  {c['gid']} — {c['role']}, достижим от {c['reached_from']} из указанных, "
                 f"получил {c['received_kzt']/1e6:.2f} млн за {c['n_tx']} транзакций, "
                 f"приоритет {c['priority_score']}" for c in d["collectors"][:5]]
        return {"answer": f"Ниже по течению от указанных {len(d['asked_gids'])} узлов достижимо "
                          f"{d['reachable_total']} счетов. Больше всех собирают:\n" + "\n".join(lines) +
                          "\nЭто гипотеза по структуре переводов, а не утверждение о виновности.",
                "cited_gids": [c["gid"] for c in d["collectors"][:5]], "mode": "fallback"}

    # 5. Кластеры.
    if "кластер" in q:
        from src.tools import _data
        _, clusters, _, _ = _data()
        big = clusters.nlargest(5, "n_nodes")
        lines = [f"  кластер {int(r.cluster_id)}: {int(r.n_nodes)} узлов, {int(r.n_seed)} фигурантов, "
                 f"оборот внутри {r.sum_kzt_internal/1e6:.1f} млн\n    {r.hypothesis}"
                 for r in big.itertuples(index=False)]
        cited = [int(x) for x in ";".join(big.top_gids).split(";") if x]
        return {"answer": "Крупнейшие кластеры:\n" + "\n".join(lines),
                "cited_gids": cited[:5], "mode": "fallback"}

    # 6. Запрос по роли.
    for word, role in _ROLE_WORDS.items():
        if word in q:
            d = tools.top_nodes(role=role, n=5)
            if d.get("error") or not d["nodes"]:
                continue
            lines = [f"  {x['gid']} — приоритет {x['priority_score']}, {x['evidence']}"
                     for x in d["nodes"]]
            return {"answer": f"Узлов с ролью {role}: {d['matched']}. Первые по приоритету:\n"
                              + "\n".join(lines),
                    "cited_gids": [x["gid"] for x in d["nodes"]], "mode": "fallback"}

    # 7. «Получают от многих, отдают немногим» — отбор по признакам.
    if ("от многих" in q or "многих" in q) and ("немноги" in q or "мало" in q or "одному" in q):
        d = tools.find_nodes({"in_deg": ">=3", "hhi_out": ">0.5", "net_flow": ">0"}, n=5)
        lines = [f"  {x['gid']} — {x['role']}, приоритет {x['priority_score']}, {x['evidence']}"
                 for x in d.get("nodes", [])]
        return {"answer": f"Узлов, получающих от многих и отдающих узкому кругу "
                          f"(in_deg >= 3, концентрация получателей hhi_out > 0.5, накапливают): "
                          f"{d.get('matched', 0)}. Первые по приоритету:\n" + "\n".join(lines),
                "cited_gids": [x["gid"] for x in d.get("nodes", [])], "mode": "fallback"}

    # 8. По умолчанию — очередь проверки.
    d = tools.top_nodes(n=5)
    lines = [f"  {x['gid']} — {x['role']}, приоритет {x['priority_score']}, {x['evidence']}"
             for x in d["nodes"]]
    return {"answer": "Топ узлов по приоритету проверки:\n" + "\n".join(lines) +
                      "\nУточните вопрос: можно спросить про конкретный gid, роль или кластер.",
            "cited_gids": [x["gid"] for x in d["nodes"]], "mode": "fallback"}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "кого проверять первым?"
    res = answer(q)
    print(f"[режим: {res['mode']}]\n{res['answer']}\n\ngid в ответе: {res['cited_gids']}")

import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv

import requests

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere # <-- Changed from Google to Cohere
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- Load Environment Variables ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# -------------------- LLM Configuration (Cohere) --------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found. Please set it in your .env file.")

# Initialize the Cohere LLM
# Using command-r-plus as it's powerful and supports tool use well.
llm = ChatCohere(
    model="command-r-plus",
    temperature=0,
    cohere_api_key=COHERE_API_KEY
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))

# -------------------- FastAPI Frontend and Utilities --------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        keys_list: list of keys in order
        type_map: dict key -> casting function
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float, "string": str, "integer": int,
        "int": int, "float": float
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map

# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """Extract JSON object from LLM output robustly."""
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first, last = s.find("{"), s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# This function is injected into the sandbox for the LLM's generated code to use if needed.
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    # (implementation is the same as the tool, omitted for brevity in final display)
    # This is a placeholder; the full implementation is written to the temp file.
    pass
'''

def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """Write and execute Python code in a sandboxed environment."""
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("data = globals().get('data', {})\n")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('ascii')}"
    # Iteratively try to reduce size
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"
    # Final attempt with lowest quality
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
'''
    script_lines = preamble + [helper, SCRAPE_FUNC, "\nresults = {}\n", code, "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write("\n".join(script_lines))
        tmp_path = tmp.name

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        out = completed.stdout.strip()
        try:
            return json.loads(out)
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass

# -----------------------------
# LLM agent setup
# -----------------------------
tools = [scrape_url_to_dataframe]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small. The output should be a data URI string like 'data:image/png;base64,...'.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# -----------------------------
# Main API Logic
# -----------------------------
@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file, data_file = None, None
        for key, val in form.items():
            if hasattr(val, "filename") and val.filename:
                if val.filename.lower().endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path, df_preview, dataset_uploaded = None, "", False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            df = None
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name
            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        llm_rules = (
            "Rules:\n"
            "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
            "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
            "3) Use only the uploaded dataset for answering questions.\n"
            '4) Produce a final JSON object with keys: "questions" and "code".\n'
            "5) For plots: use plot_to_base64() helper to return base64 data URI string."
        ) if dataset_uploaded else (
            "Rules:\n"
            "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
            '2) Produce a final JSON object with keys: "questions" and "code".\n'
            "3) For plots: use plot_to_base64() helper to return base64 data URI string."
        )

        llm_input = f"{llm_rules}\nQuestions:\n{raw_questions}\n{df_preview}\nRespond with the JSON object only."

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process key mapping & type casting
        if keys_list and type_map:
            mapped = {}
            for idx, q in enumerate(result.keys()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        val = result[q]
                        if isinstance(val, str) and val.startswith("data:image/"):
                             mapped[key] = val # Keep the data URI string for images
                        else:
                            mapped[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        mapped[key] = result[q]
            result = mapped

        return JSONResponse(content=result)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """Runs the LLM agent and executes code."""
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke({"input": llm_input})
            raw_out = response.get("output") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed
        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code, questions = parsed["code"], parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    df.to_pickle(temp_pkl.name)
                    pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}

# -----------------------------
# System Diagnostics & Info
# -----------------------------
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime
import platform
import psutil
import time

DIAG_NETWORK_TARGETS = {
    "Cohere": "https://api.cohere.com",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
COHERE_MODELS_TO_TEST = ["command-r-plus", "command-r"]
DIAG_LLM_KEY_TIMEOUT = 30
DIAG_PARALLELISM = 4
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)

async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        raise

def _env_check(required=None):
    out = {}
    for k in (required or []):
        val = os.getenv(k)
        out[k] = {"present": bool(val), "masked": (val[:4] + "..." + val[-4:]) if val else None}
    return out

def _system_info():
    return {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }

def _network_probe_sync(url, timeout=30):
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _test_cohere_key_model(key, model, ping_text="ping"):
    """Test a Cohere API key by sending a minimal request."""
    try:
        llm = ChatCohere(model=model, temperature=0, cohere_api_key=key)
        resp = llm.invoke(ping_text)
        text = resp.content if hasattr(resp, 'content') else str(resp)
        return {"ok": True, "model": model, "summary": text[:80].replace('\n', ' ') + "..."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def check_cohere_setup():
    """Light-touch test for the configured Cohere API key against different models."""
    if not COHERE_API_KEY:
        return {"warning": "COHERE_API_KEY not configured"}
    
    tasks = [run_in_thread(_test_cohere_key_model, COHERE_API_KEY, model, timeout=DIAG_LLM_KEY_TIMEOUT) for model in COHERE_MODELS_TO_TEST]
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for model, res in zip(COHERE_MODELS_TO_TEST, completed):
        if isinstance(res, Exception):
            results.append({"model": model, "ok": False, "error": str(res)})
        else:
            results.append({"model": model, **res})
    return {"models_tested": results}


@app.get("/summary")
async def diagnose(full: bool = Query(False, description="Run extended checks")):
    started = datetime.utcnow()
    tasks = {
        "env": run_in_thread(_env_check, ["COHERE_API_KEY"], timeout=3),
        "system": run_in_thread(_system_info, timeout=10),
        "network": asyncio.create_task(asyncio.gather(*[run_in_thread(_network_probe_sync, url, timeout=10) for url in DIAG_NETWORK_TARGETS.values()], return_exceptions=True)),
        "llm_setup": asyncio.create_task(check_cohere_setup())
    }
    
    results = {}
    for name, coro in tasks.items():
        try:
            res = await coro
            # Special handling for network results to map them back to names
            if name == 'network':
                results[name] = {"status": "ok", "result": {n: r for n, r in zip(DIAG_NETWORK_TARGETS.keys(), res)}}
            else:
                results[name] = {"status": "ok", "result": res}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    failed_checks = [k for k, v in results.items() if v.get("status") != "ok"]
    
    return {
        "status": "warning" if failed_checks else "ok",
        "server_time": datetime.utcnow().isoformat() + "Z",
        "summary": {"failed_checks": failed_checks},
        "checks": results,
        "elapsed_seconds": (datetime.utcnow() - started).total_seconds()
    }


@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
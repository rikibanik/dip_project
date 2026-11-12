from __future__ import annotations
import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from src.config import DB_PATH, ensure_dirs

def connect(path:Path|str=DB_PATH):
    ensure_dirs(); con=sqlite3.connect(path); con.row_factory=sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, content_hash TEXT NOT NULL, uploaded_at TEXT NOT NULL, status TEXT NOT NULL, document_type TEXT, confidence REAL, case_id TEXT NOT NULL DEFAULT 'default', result_json TEXT)")
    cols={row[1] for row in con.execute("PRAGMA table_info(documents)")}
    if "case_id" not in cols:con.execute("ALTER TABLE documents ADD COLUMN case_id TEXT NOT NULL DEFAULT 'default'")
    con.commit(); return con
@contextmanager
def _connection(path:Path|str=DB_PATH):
    con=connect(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:con.close()
def save_document(record:dict,path:Path|str=DB_PATH):
    with _connection(path) as con:
        con.execute("INSERT OR REPLACE INTO documents (document_id,filename,content_hash,uploaded_at,status,document_type,confidence,case_id,result_json) VALUES (?,?,?,?,?,?,?,?,?)",(record["document_id"],record["filename"],record.get("content_hash",""),datetime.now(timezone.utc).isoformat(),record.get("status","processed"),record.get("document_type"),record.get("classification",{}).get("confidence"),record.get("case_id","default"),json.dumps(record,ensure_ascii=False)))
def list_documents(path:Path|str=DB_PATH)->list[dict]:
    with _connection(path) as con: return [dict(row) for row in con.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")]
def get_document(document_id:str,path:Path|str=DB_PATH)->dict|None:
    with _connection(path) as con:
        row=con.execute("SELECT result_json FROM documents WHERE document_id=?",(document_id,)).fetchone()
    return json.loads(row[0]) if row else None

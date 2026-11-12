import logging
from src.config import OUTPUTS,ensure_dirs

def configure_logging()->None:
    ensure_dirs()
    logging.basicConfig(filename=OUTPUTS/"reports"/"app.log",level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",encoding="utf-8")

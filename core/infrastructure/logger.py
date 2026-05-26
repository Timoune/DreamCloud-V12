import logging
import os
 
from config.memory_config import LOG_PATH
 
os.makedirs(LOG_PATH, exist_ok=True)
 
logger = logging.getLogger("DreamCloud")
 
logger.setLevel(logging.INFO)
 
handler = logging.FileHandler(
    os.path.join(LOG_PATH, "runtime.log")
)
 
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s: %(message)s"
)
 
handler.setFormatter(formatter)
 
logger.addHandler(handler)
import logging
import logging.handlers
import os
import json
import threading
import time
import sys

# Constants
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag_mcp.log")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_config.json")
DEFAULT_LEVEL = logging.WARN

_logger_initialized = False

def setup_logging():
    global _logger_initialized
    if _logger_initialized:
        return
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create the config file if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump({"level": "WARN"}, f, indent=4)
            
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(DEFAULT_LEVEL)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Stream Handler (stderr for MCP compatibility)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    # Rotating File Handler (10MB max size, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Initial load of level
    _update_log_level_from_config()

    # Start config watcher daemon
    thread = threading.Thread(target=_config_watcher, daemon=True)
    thread.start()
    
    _logger_initialized = True


def _update_log_level_from_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                level_str = data.get("level", "WARN").upper()
                level = getattr(logging, level_str, logging.WARN)
                
                root_logger = logging.getLogger()
                if root_logger.level != level:
                    root_logger.setLevel(level)
                    root_logger.info(f"Log level dynamically updated to {level_str}")
    except Exception as e:
        # Fallback to sys.stderr so we don't cause recursive logging issues
        print(f"Error reading logging_config.json: {e}", file=sys.stderr)


def _config_watcher():
    last_mtime = 0
    while True:
        try:
            if os.path.exists(CONFIG_FILE):
                current_mtime = os.path.getmtime(CONFIG_FILE)
                if current_mtime > last_mtime:
                    _update_log_level_from_config()
                    last_mtime = current_mtime
        except Exception:
            pass
        time.sleep(5)


def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance with the standardized configuration."""
    if not _logger_initialized:
        setup_logging()
    return logging.getLogger(name)

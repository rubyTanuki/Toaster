from __future__ import annotations
import sys
import logging
from pathlib import Path
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Intercepts standard Python logging messages and routes them to Loguru.
    This prevents dependencies from leaking text into the MCP stdio stream.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the origin of the log message for accurate line numbers
        frame, depth = sys._getframe(6), 6
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def configure_cli_logging(debug: bool = False):
    """
    Configures the global logger. Call this exactly once at the entry point of your CLI/MCP server.
    """
    
    logger.remove()
    if debug:
        logger.add(sys.stdout, level="DEBUG", enqueue=True)
        logger.add(sys.stdout, level="INFO", enqueue=True)

def configure_mcp_logging(project_dir: Path | str):
    """
    Configures the global logger. Call this exactly once at the entry point of your CLI/MCP server.
    """
    project_path = Path(project_dir)
    log_dir = project_path / ".tostr"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "tostr.log"

    # Remove any existing logger outputs
    logger.remove()

    # Add the file handler with rotation and retention
    logger.add(
        str(log_file),
        rotation="5 MB",      # Create a new file when it hits 5MB
        retention="3 days",   # Automatically clean up old logs
        level="DEBUG",        # Capture everything
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {thread.name: <15} | {name}:{function}:{line} - {message}",
        enqueue=True          # Makes logging strictly thread-safe and async-safe
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("watchfiles").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
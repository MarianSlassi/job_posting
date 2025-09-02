import logging
from datetime import datetime

def get_custom_logger(config, name: str, log_file: str = None) -> logging.Logger:
    """
    Returns a configured logger that writes to both a file and the console.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Name of the log file. If not provided, the file name will include the current date and time.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        logger = get_logger("train_model")
        logger.info("Training started")
    """
    LOG_DIR = config.get('logs_dir')
    # returns ready to use logger with name from parameters
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # If folder doens't exist --> create, if no such path --> create full path with parents
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # If we didn't pass any log file directory --> create file with date and name
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOG_DIR / f"{name}_{timestamp}.log"
        else:
            log_file = LOG_DIR / log_file

        # Handler for saving in file
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Logs format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Adding handler to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

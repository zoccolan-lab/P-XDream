from zdream.utils.logger import Logger

# --- LOGGING ---

def start(logger: Logger, name: str):
    
    logger.info(mess=name)
    logger.formatting = lambda x: f'> {x}'

def end(logger: Logger):
    
    logger.info(mess='Done')
    logger.reset_formatting()
    logger.info(mess='')
import logging

logger = logging.getLogger(__name__)


def set_flags(cfg):
    # get all flags under cfg.global_flags if exists
    global_flags = cfg.get("global_flags", None)
    if global_flags is None:
        return
    for flag, value in global_flags.items():
        import xlm.flags as flags

        logger.info(f"Setting flag {flag} to {value}")
        flags.__dict__[flag] = value

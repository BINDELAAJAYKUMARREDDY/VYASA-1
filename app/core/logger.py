import logging


def get_logger(name="vyasa"):
    log = logging.getLogger(name)
    log.propagate = False

    for h in log.handlers:
        if getattr(h, "_vyasa_handler", False):
            return log

    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    h.setFormatter(fmt)
    h._vyasa_handler = True
    log.addHandler(h)
    return log


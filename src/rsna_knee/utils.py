"""Small helpers the notebook got from Jupyter."""


def display(obj):
    """Jupyter's display() inside a notebook; a plain print anywhere else."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display as _display
            return _display(obj)
    except ImportError:
        pass
    print(obj.to_string() if hasattr(obj, "to_string") else obj)

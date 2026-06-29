# Source - https://stackoverflow.com/a/279586
# Posted by Claudiu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-06, License - CC BY-SA 4.0


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

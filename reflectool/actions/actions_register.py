ACTIONS_REGISTRY = []

def register(action_name, func_type="Normal"):
    def decorator(func):
        ACTIONS_REGISTRY.append((func, func_type))
        return func
    return decorator
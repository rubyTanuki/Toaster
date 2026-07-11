from tostr.core.resolver import BaseDependencyResolver

class PythonDependencyResolver(BaseDependencyResolver):
    def __init__(self, registry: Registry):
        super().__init__(registry)
        self.strict_arity = False

    def _use_local_search(self, dep_info: tuple) -> bool:
        name, arity, receiver, is_creation = dep_info
        if receiver is None:
            return True
        bare = receiver.split('.')[0]
        return bare in ('self', 'cls')

    def _resolve_receiver_type(self, method: "BaseMethod", receiver: str) -> Optional[str]:
        from tostr.core.models import BaseClass

        # self/cls always refer to the enclosing class.
        if receiver in ('self', 'cls'):
            if isinstance(method.parent, BaseClass):
                return method.parent.uid
            return None

        # self.field or cls.field — look up the first-level field type in the parent class.
        if receiver.startswith(('self.', 'cls.')):
            field_name = receiver.split('.', 1)[1].split('.')[0]
            if isinstance(method.parent, BaseClass):
                for f in method.parent.fields:
                    if f.name == field_name and f.field_type:
                        return f.field_type
            return None

        return super()._resolve_receiver_type(method, receiver)

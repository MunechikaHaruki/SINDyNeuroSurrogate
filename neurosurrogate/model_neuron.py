class Compartment:
    def __init__(
        self, gate_inits: list[float], gate_names: list[str], v_init: float = -65
    ):
        self.v_init = v_init
        self.gate_inits = gate_inits
        self.gate_names = gate_names

    @property
    def vars(self):
        return ["V"] + self.gate_names

    @property
    def gate(self):
        return [False] + [True] * len(self.gate_names)

    @property
    def init(self):
        return [self.v_init] + self.gate_inits

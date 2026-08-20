from .hh import HH_TYPE, PASSIVE_TYPE
from .traub import TRAUB_TYPE

# type 名文字列 → CompartmentType の dispatch table (from_dict / chain 等で使用)
COMPARTMENT_TYPES = {
    "hh": HH_TYPE,
    "passive": PASSIVE_TYPE,
    "traub": TRAUB_TYPE,
}

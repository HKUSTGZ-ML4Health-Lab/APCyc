"""Historical model exports without an APCyc import cycle."""
from .autoencoder.model import AutoEncoder
from .LDM.ldm import LDMPepDesign


def __getattr__(name):
    if name == 'APCycLDMPepDesign':
        from apcyc.model import APCycLDMPepDesign
        return APCycLDMPepDesign
    raise AttributeError(name)

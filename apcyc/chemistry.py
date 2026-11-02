"""Paper residue compatibility shared by topology heads and output inspection."""


def is_iso_pair(first, second):
    """Accept canonical one-letter or PDB three-letter residue names."""
    names = {'LYS': 'K', 'ASP': 'D', 'ASN': 'N'}
    first, second = names.get(first, first), names.get(second, second)
    return (first == 'K' and second in {'D', 'N'}) or (second == 'K' and first in {'D', 'N'})

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple

from data.format import VOCAB
from utils import const

_CYC_CFG: Dict = {}
_TOKEN_MAP: Dict[str, Dict] = {}


def _base_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    base_symbol_to_abrv = {symbol: abrv for symbol, abrv in const.aas}
    base_abrv_to_symbol = {abrv: symbol for symbol, abrv in const.aas}
    return base_symbol_to_abrv, base_abrv_to_symbol


def _resolve_symbol_key(key: str, base_abrv_to_symbol: Dict[str, str]) -> Optional[str]:
    if not key:
        return None
    key = key.upper()
    if len(key) == 1 and key in base_abrv_to_symbol.values():
        return key
    if key in base_abrv_to_symbol:
        return base_abrv_to_symbol[key]
    return None


def _ensure_vocab_token(symbol: str, abrv: str, base_symbol: str, base_abrv: str) -> None:
    if symbol in VOCAB.symbol2idx:
        if hasattr(VOCAB, 'register_alias'):
            VOCAB.register_alias(symbol, base_symbol, base_abrv)
        else:
            if hasattr(VOCAB, 'alias_symbol'):
                VOCAB.alias_symbol[symbol] = base_symbol
            if hasattr(VOCAB, 'alias_abrv'):
                VOCAB.alias_abrv[symbol] = base_abrv
        return
    idx = len(VOCAB.idx2block)
    VOCAB.idx2block.append((symbol, abrv))
    VOCAB.symbol2idx[symbol] = idx
    VOCAB.abrv2idx[abrv] = idx
    VOCAB.special_mask.append(0)
    if hasattr(VOCAB, 'register_alias'):
        VOCAB.register_alias(symbol, base_symbol, base_abrv)
    else:
        if hasattr(VOCAB, 'alias_symbol'):
            VOCAB.alias_symbol[symbol] = base_symbol
        if hasattr(VOCAB, 'alias_abrv'):
            VOCAB.alias_abrv[symbol] = base_abrv
    const.sidechain_atoms[symbol] = const.sidechain_atoms.get(base_symbol, [])
    const.sidechain_bonds[symbol] = const.sidechain_bonds.get(base_symbol, [])


def _resolve_symbol(symbol: str) -> str:
    if hasattr(VOCAB, 'resolve_symbol'):
        return VOCAB.resolve_symbol(symbol)
    if hasattr(VOCAB, 'alias_symbol'):
        return VOCAB.alias_symbol.get(symbol, symbol)
    return symbol


def _build_minimal_vocab_defs() -> Tuple[List[Dict], Dict[str, Dict]]:
    base_symbol_to_abrv, _ = _base_maps()
    token_defs = []
    token_map: Dict[str, Dict] = {'SS': {}, 'ISO': {}, 'HT_N': {}, 'HT_C': {}}

    # SS
    token_defs.append({
        'symbol': 'CYS_SSB',
        'abrv': 'CYS_SSB',
        'base_symbol': 'C',
        'base_abrv': base_symbol_to_abrv['C'],
    })
    token_map['SS']['C'] = 'CYS_SSB'

    # ISO
    iso_map = {'K': 'LYS_ISO', 'D': 'ASP_ISO', 'E': 'GLU_ISO'}
    for base_symbol, symbol in iso_map.items():
        token_defs.append({
            'symbol': symbol,
            'abrv': symbol,
            'base_symbol': base_symbol,
            'base_abrv': base_symbol_to_abrv[base_symbol],
        })
        token_map['ISO'][base_symbol] = symbol

    # HT (per-residue tokens)
    for base_symbol, base_abrv in base_symbol_to_abrv.items():
        n_sym = f'{base_abrv}_NCYC'
        c_sym = f'{base_abrv}_CCYC'
        token_defs.append({
            'symbol': n_sym,
            'abrv': n_sym,
            'base_symbol': base_symbol,
            'base_abrv': base_abrv,
        })
        token_defs.append({
            'symbol': c_sym,
            'abrv': c_sym,
            'base_symbol': base_symbol,
            'base_abrv': base_abrv,
        })
        token_map['HT_N'][base_symbol] = n_sym
        token_map['HT_C'][base_symbol] = c_sym

    return token_defs, token_map


def _build_custom_vocab_defs(custom_map: Dict) -> Tuple[List[Dict], Dict[str, Dict]]:
    base_symbol_to_abrv, base_abrv_to_symbol = _base_maps()
    token_defs = []
    token_map: Dict[str, Dict] = {'SS': {}, 'ISO': {}, 'HT_N': {}, 'HT_C': {}}

    for section, mapping in custom_map.items():
        if not isinstance(mapping, dict):
            continue
        section_key = section.upper()
        for base_key, symbol in mapping.items():
            base_symbol = _resolve_symbol_key(str(base_key), base_abrv_to_symbol)
            if base_symbol is None:
                continue
            base_abrv = base_symbol_to_abrv[base_symbol]
            token_defs.append({
                'symbol': symbol,
                'abrv': symbol,
                'base_symbol': base_symbol,
                'base_abrv': base_abrv,
            })
            if section_key in token_map:
                token_map[section_key][base_symbol] = symbol

    return token_defs, token_map


def apply_enlarged_vocab(cyc_cfg: Dict) -> None:
    if not cyc_cfg or not cyc_cfg.get('enable_enlarged_vocab', False):
        return
    vocab_mode = cyc_cfg.get('vocab_mode', 'minimal')
    vocab_custom_map = cyc_cfg.get('vocab_custom_map', None)

    if vocab_mode == 'custom' and isinstance(vocab_custom_map, dict):
        token_defs, token_map = _build_custom_vocab_defs(vocab_custom_map)
    else:
        token_defs, token_map = _build_minimal_vocab_defs()

    for token in token_defs:
        _ensure_vocab_token(
            token['symbol'],
            token['abrv'],
            token['base_symbol'],
            token['base_abrv'],
        )

    _TOKEN_MAP.clear()
    _TOKEN_MAP.update(token_map)


def set_config(cyc_cfg: Dict) -> None:
    _CYC_CFG.clear()
    _CYC_CFG.update(cyc_cfg or {})


def get_config() -> Dict:
    return _CYC_CFG


def get_token_map() -> Dict[str, Dict]:
    return _TOKEN_MAP


def rewrite_tokens_by_cyclization(S: List[int], mask: List[int], record: Dict, strict: bool = False) -> List[int]:
    cfg = get_config()
    if not cfg.get('enable_enlarged_vocab', False):
        return S
    token_map = get_token_map()
    if not token_map:
        return S
    if not record or 'c_star' not in record or 'gt_pair' not in record:
        return S
    c_star = record['c_star']
    gt_pair = record['gt_pair']
    if not isinstance(gt_pair, (list, tuple)) or len(gt_pair) < 2:
        return S

    pep_offset = mask.count(0)
    i_global = pep_offset + int(gt_pair[0])
    j_global = pep_offset + int(gt_pair[1])
    if i_global >= len(S) or j_global >= len(S):
        return S

    def _base_symbol_at(idx: int) -> str:
        symbol = VOCAB.idx_to_symbol(int(S[idx]))
        return _resolve_symbol(symbol)

    def _apply_token(idx: int, symbol_map: Dict[str, str]) -> None:
        base_symbol = _base_symbol_at(idx)
        if base_symbol not in symbol_map:
            if strict:
                raise ValueError(f'No vocab mapping for {base_symbol} at {idx}')
            return
        S[idx] = VOCAB.symbol_to_idx(symbol_map[base_symbol])

    if c_star == 'SS':
        if 'SS' in token_map:
            _apply_token(i_global, token_map['SS'])
            _apply_token(j_global, token_map['SS'])
    elif c_star == 'ISO':
        if 'ISO' in token_map:
            _apply_token(i_global, token_map['ISO'])
            _apply_token(j_global, token_map['ISO'])
    elif c_star == 'HT':
        if 'HT_N' in token_map and 'HT_C' in token_map:
            _apply_token(i_global, token_map['HT_N'])
            _apply_token(j_global, token_map['HT_C'])
    return S

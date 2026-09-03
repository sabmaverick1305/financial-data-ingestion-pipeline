"""FIES Parser Engine — a parser-agnostic execution boundary for document parsing.

Downstream code should depend on `fies_parser.engine` and `fies_parser.canonical`
only. Concrete adapters (`fies_parser.adapters.*`) may import third-party parser
libraries; nothing else in this package or its callers should.
"""

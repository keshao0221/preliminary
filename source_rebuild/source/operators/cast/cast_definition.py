"""NineToothed Cast: conversion is defined by the output tensor's dtype."""
import ninetoothed

RANK = 8
TILE = 256

def arrangement(input, output):
    tile = (1,) * (RANK - 1) + (TILE,)
    return input.tile(tile), output.tile(tile)

def application(input, output):
    output = input

application.__annotations__ = dict(zip(('input', 'output'), arrangement(ninetoothed.Tensor(RANK), ninetoothed.Tensor(RANK))))

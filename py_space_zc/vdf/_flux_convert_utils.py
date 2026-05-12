#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def broadcast_energy_to_data(energy, data, dims=None):
    """Return energy reshaped so it broadcasts to distribution data.

    Supported common cases:
    - energy shape (energy,), data shape (time, energy)
    - energy shape (energy,), data shape (time, energy, pitchangle)
    - energy shape (time, energy), data shape (time, energy, pitchangle)
    - energy shape (time, energy), data shape (time, energy, phi, theta)
    """
    energy = np.asarray(energy, dtype=float)
    data = np.asarray(data)
    dims = tuple(dims) if dims is not None else None

    if energy.ndim == 0:
        return energy

    if energy.ndim == 1:
        energy_axis = _find_energy_axis(energy.size, data.shape, dims)
        shape = [1] * data.ndim
        shape[energy_axis] = energy.size
        return energy.reshape(shape)

    if energy.ndim == 2:
        time_axis, energy_axis = _find_time_energy_axes(energy.shape, data.shape, dims)
        shape = [1] * data.ndim
        shape[time_axis] = energy.shape[0]
        shape[energy_axis] = energy.shape[1]
        return energy.reshape(shape)

    if energy.shape == data.shape:
        return energy

    raise ValueError("energy must be scalar, 1D, 2D, or have the same shape as data.")


def _find_energy_axis(n_energy, data_shape, dims):
    if dims is not None:
        for name in ("energy", "energy_eV", "e", "E", "idx0"):
            if name in dims and data_shape[dims.index(name)] == n_energy:
                return dims.index(name)

    matches = [i for i, n in enumerate(data_shape) if n == n_energy]
    if not matches:
        raise ValueError("Could not match 1D energy length to any data dimension.")
    if len(matches) == 1:
        return matches[0]
    if len(data_shape) > 1 and data_shape[1] == n_energy:
        return 1
    return matches[0]


def _find_time_energy_axes(energy_shape, data_shape, dims):
    if dims is not None:
        time_axis = None
        energy_axis = None
        for name in ("time", "epoch"):
            if name in dims and data_shape[dims.index(name)] == energy_shape[0]:
                time_axis = dims.index(name)
                break
        for name in ("energy", "energy_eV", "e", "E", "idx0"):
            if name in dims and data_shape[dims.index(name)] == energy_shape[1]:
                energy_axis = dims.index(name)
                break
        if time_axis is not None and energy_axis is not None:
            return time_axis, energy_axis

    if len(data_shape) >= 2 and data_shape[0] == energy_shape[0] and data_shape[1] == energy_shape[1]:
        return 0, 1

    for i, ni in enumerate(data_shape):
        if ni != energy_shape[0]:
            continue
        for j, nj in enumerate(data_shape):
            if i != j and nj == energy_shape[1]:
                return i, j

    raise ValueError("Could not match 2D energy shape to data dimensions.")

"""Saves one spectrum per HDF5 file, laid out as a NeXus NXdata group."""

import os
from dataclasses import asdict

import h5py


def make_filename(template, path, name, number):
    """Expand an areaDetector-style template such as ``"%s%s_%04d.h5"``."""
    return template % (path, name, number)


def write_spectrum(filename, spectrum, region, stats):
    """
    Write ``spectrum`` to ``filename``, with the ``region`` settings and the
    ``stats`` dict as attributes. Creates the directory if needed.
    """
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with h5py.File(filename, "w-") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "image"
        data.attrs["axes"] = ["y", "energy"]
        data.create_dataset("image", data=spectrum.data, compression="gzip")
        data.create_dataset("energy", data=spectrum.x_axis.array())
        data.create_dataset("y", data=spectrum.y_axis.array())
        data.create_dataset("edc", data=spectrum.data.sum(axis=0))
        region_group = entry.create_group("region")
        for key, value in asdict(region).items():
            if value is not None:
                region_group.attrs[key] = value
        stats_group = entry.create_group("stats")
        for key, value in stats.items():
            stats_group.attrs[key] = value
        for key, value in spectrum.metadata.items():
            entry.attrs[key] = value

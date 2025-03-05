import numpy as np
import pytest

import hoomd
from hoomd.logging import LoggerCategories
from hoomd.conftest import operation_pickling_check, logging_check
from hoomd import md

from hoomd.logging import LoggerCategories


def test_attaching(simulation_factory, two_particle_snapshot_factory):

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=0.5))
    integrator = md.Integrator(dt=0.005)

    thermostat= hoomd.md.methods.thermostats.MTTK(kT=1,tau=0.5)
    nvt = hoomd.md.methods.ConstantVolumeSLLOD(
        filter=hoomd.filter.All(), thermostat=thermostat, shear_rate=0.1
    )

    integrator.methods.append(nvt)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), 1.0)
    sim.run(0)


def test_run(simulation_factory, two_particle_snapshot_factory):



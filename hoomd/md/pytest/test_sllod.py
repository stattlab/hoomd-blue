import numpy as np
import pytest

import hoomd
from hoomd.logging import LoggerCategories
from hoomd.conftest import operation_pickling_check, logging_check
from hoomd import md
from hoomd.logging import LoggerCategories

import numpy

def test_attaching(simulation_factory, two_particle_snapshot_factory):

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=0.5))
    integrator = md.Integrator(dt=0.005)

    thermostat= hoomd.md.methods.thermostats.MTTK(kT=1,tau=0.5)
    sllod = hoomd.md.methods.ConstantVolumeSLLOD(
        filter=hoomd.filter.All(), thermostat=thermostat, shear_rate=0.1)

    integrator.methods.append(sllod)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), 1.0)
    sim.run(0)

def test_box_deform(simulation_factory, two_particle_snapshot_factory):

    L = 30
    shear_rate = 0.2
    timestep = 0.005
    sim = simulation_factory(two_particle_snapshot_factory(L=L))
    integrator = md.Integrator(dt=timestep)

    thermostat= hoomd.md.methods.thermostats.MTTK(kT=1,tau=0.5)
    sllod = hoomd.md.methods.ConstantVolumeSLLOD(
        filter=hoomd.filter.All(), thermostat=thermostat, shear_rate=shear_rate)

    integrator.methods.append(sllod)
    # box tilt = L*shear_rate*dt
    # L0*erate*dt

    sim.operations.integrator = integrator
    snap =sim.state.get_snapshot()
    # cubic box before
    numpy.testing.assert_allclose(snap.configuration.box[0],L)
    numpy.testing.assert_allclose(snap.configuration.box[1],L)
    numpy.testing.assert_allclose(snap.configuration.box[2],L)
    numpy.testing.assert_allclose(snap.configuration.box[3],0)
    numpy.testing.assert_allclose(snap.configuration.box[4],0)
    numpy.testing.assert_allclose(snap.configuration.box[5],0)

    sim.run(1)
    snap =sim.state.get_snapshot()
    numpy.testing.assert_allclose(snap.configuration.box[0],L)
    numpy.testing.assert_allclose(snap.configuration.box[1],L)
    numpy.testing.assert_allclose(snap.configuration.box[2],L)
    numpy.testing.assert_allclose(snap.configuration.box[3],shear_rate*timestep)
    numpy.testing.assert_allclose(snap.configuration.box[4],0)
    numpy.testing.assert_allclose(snap.configuration.box[5],0)

    sim.run(5) # has run 6 timesteps by now
    snap =sim.state.get_snapshot()
    numpy.testing.assert_allclose(snap.configuration.box[3],6*shear_rate*timestep)


def test_pbc_velocity_correction(simulation_factory, two_particle_snapshot_factory):

    shear_rate = 0.25
    timestep = 0.005
    velocity = 10
    L = 20

    snapshot = two_particle_snapshot_factory(dimensions=3, d=1, L=L)
    if snapshot.communicator.rank == 0:
        snapshot.particles.position[0] = (0,9.99,0)
        snapshot.particles.position[1] = (0,-9.99,0)
        snapshot.particles.velocity[0] = (0,velocity,0)
        snapshot.particles.velocity[1] = (0,-velocity,0)


    sim = simulation_factory(snapshot)
    integrator = hoomd.md.Integrator(dt=timestep)

    sllod = hoomd.md.methods.ConstantVolumeSLLOD(
        filter=hoomd.filter.All(), thermostat=None, shear_rate=shear_rate)


    integrator.methods.append(sllod)

    sim.operations.integrator = integrator
    sim.run(1)
    snap =sim.state.get_snapshot()

    numpy.testing.assert_allclose(snap.particles.velocity[0][0], -L*shear_rate-velocity*shear_rate*timestep)
    numpy.testing.assert_allclose(snap.particles.velocity[1][0], L*shear_rate-(-velocity)*shear_rate*timestep)


def test_sllod_correction(simulation_factory, two_particle_snapshot_factory):

    shear_rate = 0.25
    timestep = 0.005
    velocity = 10
    L = 10
    pos = 3

    snapshot = two_particle_snapshot_factory(dimensions=3, d=1, L=L)
    if snapshot.communicator.rank == 0:
        snapshot.particles.position[0] = (0,pos,0)
        snapshot.particles.position[1] = (0,-pos,0)
        snapshot.particles.velocity[0] = (0,velocity,0)
        snapshot.particles.velocity[1] = (0,velocity,0)


    sim = simulation_factory(snapshot)
    integrator = hoomd.md.Integrator(dt=timestep)

    sllod = hoomd.md.methods.ConstantVolumeSLLOD(
        filter=hoomd.filter.All(), thermostat=None, shear_rate=shear_rate)

    integrator.methods.append(sllod)

    sim.operations.integrator = integrator
    sim.run(1)
    snap =sim.state.get_snapshot()

    numpy.testing.assert_allclose(snap.particles.velocity[0][0],-shear_rate*velocity*timestep)
    numpy.testing.assert_allclose(snap.particles.velocity[1][0],-shear_rate*velocity*timestep)


def test_thermo_compute(simulation_factory, two_particle_snapshot_factory):

    thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())

    integrator = hoomd.md.Integrator(dt=0.0001)
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    integrator.methods.append(hoomd.md.methods.ConstantVolumeSLLOD(hoomd.filter.All(), thermostat,shear_rate=0.1))

    snapshot = two_particle_snapshot_factory(dimensions=3, d=1, L=10)
    if snapshot.communicator.rank == 0:
        snapshot.particles.velocity[0] = (0.0,0.0,0.5)
        snapshot.particles.velocity[1] = (0.5,0.0,0.0)

    sim = simulation_factory(snapshot)
    sim.operations.integrator = integrator
    sim.operations.add(thermo)
    sim.run(1)

    np.testing.assert_allclose(thermo.kinetic_energy, 0.25)
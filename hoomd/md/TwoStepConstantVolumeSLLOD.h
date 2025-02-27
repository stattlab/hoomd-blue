// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTVOLUME_SLLOD_H
#define HOOMD_TWOSTEPCONSTANTVOLUME_SLLOD_H

#include "ComputeThermo.h"
#include "TwoStepConstantVolume.h"
#include "Thermostat.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
namespace hoomd::md
    {

/** Perform constant volume simulation.

    Implement the the Velocity-Verlet integration scheme with an optional velocity rescaling
    Thermostat.
*/
class PYBIND11_EXPORT TwoStepConstantVolumeSLLOD : public TwoStepConstantVolume
    {
    public:
    /** Construct the constant volume integration method.

        @param sysdef System to work on.
        @param group Subset of particles to integrate.
        @param thermostat Thermostat to use. Set to null for constant energy simulations.
    */
    TwoStepConstantVolumeSLLOD(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<Thermostat> thermostat,
                          Scalar shear_rate)
        : TwoStepConstantVolume(sysdef, group, thermostat), m_shear_rate(shear_rate)
        {
            BoxDim global_box = m_pdata->getGlobalBox();
            const Scalar3 global_hi = global_box.getHi();
            m_boundary_shear_velocity = global_hi.y * m_shear_rate*2;
        }

    virtual ~TwoStepConstantVolumeSLLOD() { }

    /** Performs the first half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per
        the Velocity-Verlet method.
    */
    virtual void integrateStepOne(uint64_t timestep);

    /** Performs the second half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle velocities are moved forward to timestep+1.
    */
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    bool deformGlobalBox();

    Scalar m_shear_rate;
    Scalar m_boundary_shear_velocity;
    };

    } // namespace hoomd::md

#endif // HOOMD_TWOSTEPCONSTANTVOLUME_H

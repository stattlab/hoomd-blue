// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTVOLUME_SLLOD_GPU_H
#define HOOMD_TWOSTEPCONSTANTVOLUME_SLLOD_GPU_H

#include "Thermostat.h"
#include "TwoStepConstantVolume.h"
#include <hoomd/Autotuner.h>

namespace hoomd::md
    {

/// Implement TwoStepConstantVolume on the GPU.
class PYBIND11_EXPORT TwoStepConstantVolumeSLLODGPU : public TwoStepConstantVolumeSLLOD
    {
    public:
    TwoStepConstantVolumeSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<Thermostat> thermostat,
                             Scalar shear_rate);

    virtual ~TwoStepConstantVolumeSLLODGPU() { }

    virtual void integrateStepOne(uint64_t timestep);

    virtual void integrateStepTwo(uint64_t timestep);

    protected:

    };
    } // namespace hoomd::md
#endif // HOOMD_TWOSTEPCONSTANTVOLUMEGPU_H

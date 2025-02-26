// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermo.h"

/*! \file ComputeThermoGPU.h
    \brief Declares a class for computing thermodynamic quantities on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_SLLOD_GPU_H__
#define __COMPUTE_THERMO_SLLOD_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes thermodynamic properties of a group of particles on the GPU
/*! ComputeThermoGPU is a GPU accelerated implementation of ComputeThermo
    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoSLLODGPU : public ComputeThermoSLLOD
    {
    public:
    //! Constructs the compute
    ComputeThermoSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group);
    virtual ~ComputeThermoSLLODGPU();

    //! Does the actual computation
    virtual void computeProperties();

    };

    } // end namespace md
    } // end namespace hoomd

#endif

// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "SmoothedHarmonicDihedralForceCompute.h"
#include "SmoothedHarmonicDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file HarmonicDihedralForceComputeGPU.h
    \brief Declares the HarmonicDihedralForceGPU class
*/

#ifndef __SMOOTHEDHARMONICDIHEDRALFORCECOMPUTEGPU_H__
#define __SMOOTHEDHARMONICDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the harmonic dihedral force calculation on the GPU
/*! HarmonicDihedralForceComputeGPU implements the same calculations as
   HarmonicDihedralForceCompute, but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in dihedralforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT SmoothedHarmonicDihedralForceComputeGPU : public SmoothedHarmonicDihedralForceCompute
    {
    public:
    //! Constructs the compute
    SmoothedHarmonicDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> system);
    //! Destructor
    ~SmoothedHarmonicDihedralForceComputeGPU();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0, Scalar cos_on, Scalar cos_cut, Scalar Vs, int m);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params;            //!< Parameters stored on the GPU (k,sign,m)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif

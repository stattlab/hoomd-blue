// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "UpdaterCylinderRotate.h"
#include "hoomd/RNGIdentifiers.h"
#include <cmath>

namespace hoomd
    {
namespace hpmc
    {
UpdaterCylinderRotate::UpdaterCylinderRotate(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<Trigger> trigger,
                                           std::shared_ptr<IntegratorHPMC> mc,
                                           double max_overlaps_per_particle,
                                           double g)
    : Updater(sysdef, trigger), m_mc(mc), m_max_overlaps_per_particle(max_overlaps_per_particle), m_g(0.01)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterCylinderRotate" << std::endl;
    setG(g);

    // allocate memory for m_pos_backup
    //unsigned int MaxN = m_pdata->getMaxN();
    //GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    //m_pdata->getMaxParticleNumberChangeSignal()
    //    .connect<UpdaterCylinderRotate, &UpdaterCylinderRotate::slotMaxNChange>(this);

    m_last_move_counters = m_mc->getCounters();
    }

UpdaterCylinderRotate::~UpdaterCylinderRotate()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterCylinderRotate" << std::endl;
    //m_pdata->getMaxParticleNumberChangeSignal()
    //    .disconnect<UpdaterCylinderRotate, &UpdaterCylinderRotate::slotMaxNChange>(this);
    }

void UpdaterCylinderRotate::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "UpdaterCylinderRotate: " << timestep << std::endl;

    // count the number of overlaps in the current configuration
    auto n_overlaps = m_mc->countOverlaps(false);
    BoxDim current_box = m_pdata->getGlobalBox();
    quat<Scalar> current_m_aq = current_box.getAlphaQuat();
    //BoxDim target_box = BoxDim((*m_target_box)(timestep));
    if (n_overlaps == 0){
        performAlphaMove(timestep, current_m_aq, m_g);
    }

    m_is_complete = false;
    }

//template<unsigned int dim, class RNG>
void UpdaterCylinderRotate::performAlphaMove(uint64_t timestep, quat<Scalar>& m_aq, Scalar g)
    {
        BoxDim old_box = m_pdata->getGlobalBox();
        auto current_m_aq = old_box.getAlphaQuat();

        // Create a prng instance for this timestep
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::UpdaterCylinderRotate, timestep, m_sysdef->getSeed()),
            hoomd::Counter(m_instance));
        g /= Scalar(2.0);
        Scalar gamma = hoomd::UniformDistribution<Scalar>(-g, g)(rng);
        quat<Scalar> q(fast::cos(gamma),
                    fast::sin(gamma)
                        * vec3<Scalar>(Scalar(0), Scalar(0), Scalar(1))); // rotation quaternion
        quat<Scalar> new_m_aq = current_m_aq * q;
        new_m_aq = new_m_aq * (fast::rsqrt(norm2(new_m_aq)));

        BoxDim new_box = old_box;
        new_box.setAlphaQuat(new_m_aq);

        m_pdata->setGlobalBox(new_box);
        auto n_overlaps = m_mc->countOverlaps(false);

        if (n_overlaps > m_max_overlaps_per_particle * m_pdata->getNGlobal())
            {
                old_box.setAlphaQuat(current_m_aq);
                m_pdata->setGlobalBox(old_box);
                //Unsure if I need to use the next line
                m_mc->communicate(false);
            }

    }

namespace detail
    {
void export_UpdaterCylinderRotate(pybind11::module& m)
    {
    pybind11::class_<UpdaterCylinderRotate, Updater, std::shared_ptr<UpdaterCylinderRotate>>(
        m,
        "UpdaterCylinderRotate")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMC>,
                            double,
                            double>()
                            )
        .def("isComplete", &UpdaterCylinderRotate::isComplete)
        .def_property("max_overlaps_per_particle",
                      &UpdaterCylinderRotate::getMaxOverlapsPerParticle,
                      &UpdaterCylinderRotate::setMaxOverlapsPerParticle)
        .def_property("g",
                      &UpdaterCylinderRotate::getG,
                      &UpdaterCylinderRotate::setG)
        .def_property("instance",
                      &UpdaterCylinderRotate::getInstance,
                      &UpdaterCylinderRotate::setInstance);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

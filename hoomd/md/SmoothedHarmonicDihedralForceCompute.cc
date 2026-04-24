// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "SmoothedHarmonicDihedralForceCompute.h"
#include "hoomd/HOOMDMath.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>


using namespace std;

/*! \file SmoothedHarmonicDihedralForceCompute.cc
    \brief Contains code for the SmoothedHarmonicDihedralForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
SmoothedHarmonicDihedralForceCompute::SmoothedHarmonicDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_K(NULL), m_sign(NULL), m_multi(NULL), m_phi_0(NULL), m_cos_on(NULL), m_cos_cut(NULL), m_VS(NULL), m_m(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing SmoothedHarmonicDihedralForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    // allocate the parameters
    m_K = new Scalar[m_dihedral_data->getNTypes()];
    m_sign = new Scalar[m_dihedral_data->getNTypes()];
    m_multi = new int[m_dihedral_data->getNTypes()];
    m_phi_0 = new Scalar[m_dihedral_data->getNTypes()];
    m_cos_on = new Scalar[m_dihedral_data->getNTypes()];
    m_cos_cut = new Scalar[m_dihedral_data->getNTypes()];
    m_VS = new Scalar[m_dihedral_data->getNTypes()];
    m_m = new int[m_dihedral_data->getNTypes()];
    }

SmoothedHarmonicDihedralForceCompute::~SmoothedHarmonicDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying SmoothedHarmonicDihedralForceCompute" << endl;

    delete[] m_K;
    delete[] m_sign;
    delete[] m_multi;
    delete[] m_phi_0;
    delete[] m_cos_on;
    delete[] m_cos_cut;
    delete[] m_VS;
    delete[] m_m;

    m_K = NULL;
    m_sign = NULL;
    m_multi = NULL;
    m_phi_0 = NULL;
    m_cos_on = NULL;
    m_cos_cut = NULL;
    m_VS = NULL;
    m_m = NULL;
    }

/*! \param type Type of the dihedral to set parameters for
    \param K Stiffness parameter for the force computation
    \param sign the sign of the cosign term
    \param multiplicity of the dihedral itself

    Sets parameters for the potential of a particular dihedral type
*/
void SmoothedHarmonicDihedralForceCompute::setParams(unsigned int type,
                                             Scalar K,
                                             Scalar sign,
                                             int multiplicity,
                                             Scalar phi_0,
                                             Scalar cos_on,
                                             Scalar cos_cut,
                                             Scalar VS,
                                             int m)
    {
    // make sure the type is valid
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    m_K[type] = K;
    m_sign[type] = sign;
    m_multi[type] = multiplicity;
    m_phi_0[type] = phi_0;
    m_cos_on[type] = cos_on;
    m_cos_cut[type] = cos_cut;
    m_VS[type] = VS;
    m_m[type] = m;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified K <= 0" << endl;
    if (sign != 1 && sign != -1)
        m_exec_conf->msg->warning()
            << "dihedral.harmonic: a non unitary sign was specified" << endl;
    if (phi_0 < 0 || phi_0 >= 2 * M_PI)
        m_exec_conf->msg->warning()
            << "dihedral.harmonic: specified phi_0 outside [0, 2pi)" << endl;

    if (cos_on<= 90 || cos_on>180)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified smoothing on angle <= 90 or > 180." << endl;
    if (cos_cut<=90 || cos_cut>180)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified smoothing cut angle <= 90 or > 180." << endl;
    if (cos_cut < cos_on)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified smoothing cut angle < on angle." << endl;
    if (m < 2)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified smoothing polynominal m < 2." << endl;
    }

void SmoothedHarmonicDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    // make sure the type is valid
    auto typ = m_dihedral_data->getTypeByName(type);
    dihedral_smoothed_harmonic_params _params(params);
    setParams(typ, _params.k, _params.d, _params.n, _params.phi_0, _params.cos_on, _params.cos_cut, _params.Vs, _params.m);
    }

pybind11::dict SmoothedHarmonicDihedralForceCompute::getParams(std::string type)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["d"] = m_sign[typ];
    params["n"] = m_multi[typ];
    params["phi0"] = m_phi_0[typ];
    params["smoothing_on"] = m_cos_on[typ];
    params["smoothing_cut"] = m_cos_cut[typ];
    params["smoothing_Vs"] = m_VS[typ];
    params["smoothing_m"] = m_m[typ];

    return params;
    }


/*! Actually perform the force computation
    \param timestep Current time step
 */
void SmoothedHarmonicDihedralForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // Zero data for force calculation.
    m_force.zeroFill();
    m_virial.zeroFill();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // for each of the dihedrals
    const unsigned int size = (unsigned int)m_dihedral_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(i);
        assert(dihedral.tag[0] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[1] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[2] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[3] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[dihedral.tag[0]];
        unsigned int idx_b = h_rtag.data[dihedral.tag[1]];
        unsigned int idx_c = h_rtag.data[dihedral.tag[2]];
        unsigned int idx_d = h_rtag.data[dihedral.tag[3]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.harmonic: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1]
                << " " << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x;
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y;
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;

        dcbm = box.minImage(dcbm);

        Scalar aax = dab.y * dcbm.z - dab.z * dcbm.y;
        Scalar aay = dab.z * dcbm.x - dab.x * dcbm.z;
        Scalar aaz = dab.x * dcbm.y - dab.y * dcbm.x;

        Scalar bbx = ddc.y * dcbm.z - ddc.z * dcbm.y;
        Scalar bby = ddc.z * dcbm.x - ddc.x * dcbm.z;
        Scalar bbz = ddc.x * dcbm.y - ddc.y * dcbm.x;

        Scalar raasq = aax * aax + aay * aay + aaz * aaz;
        Scalar rbbsq = bbx * bbx + bby * bby + bbz * bbz;
        Scalar rgsq = dcbm.x * dcbm.x + dcbm.y * dcbm.y + dcbm.z * dcbm.z;
        Scalar rg = sqrt(rgsq);

        Scalar rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = Scalar(0.0);
        if (rg > Scalar(0.0))
            rginv = Scalar(1.0) / rg;
        if (raasq > Scalar(0.0))
            raa2inv = Scalar(1.0) / raasq;
        if (rbbsq > Scalar(0.0))
            rbb2inv = Scalar(1.0) / rbbsq;
        Scalar rabinv = sqrt(raa2inv * rbb2inv);

        Scalar c_abcd = (aax * bbx + aay * bby + aaz * bbz) * rabinv;
        Scalar s_abcd = rg * rabinv * (aax * ddc.x + aay * ddc.y + aaz * ddc.z);

        if (c_abcd > 1.0)
            c_abcd = 1.0;
        if (c_abcd < -1.0)
            c_abcd = -1.0;

        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
        int multi = m_multi[dihedral_type];
        Scalar p = Scalar(1.0);
        Scalar dfab = Scalar(0.0);
        Scalar ddfab = Scalar(0.0);

        for (int j = 0; j < multi; j++)
            {
            ddfab = p * c_abcd - dfab * s_abcd;
            dfab = p * s_abcd + dfab * c_abcd;
            p = ddfab;
            }

        /////////////////////////
        // FROM LAMMPS: sin_shift is always 0... so dropping all sin_shift terms!!!!
        // Adding charmm dihedral functionality, sin_shift not always 0,
        // cos_shift not always 1
        /////////////////////////

        Scalar sign = m_sign[dihedral_type];
        Scalar phi_0 = m_phi_0[dihedral_type];
        Scalar sin_phi_0 = fast::sin(phi_0);
        Scalar cos_phi_0 = fast::cos(phi_0);
        p = p * cos_phi_0 + dfab * sin_phi_0;
        p = p * sign;
        dfab = dfab * cos_phi_0 - ddfab * sin_phi_0;
        dfab = dfab * sign;
        dfab *= (Scalar)-multi;
        p += Scalar(1.0);

        if (multi == 0)
            {
            p = Scalar(1.0) + sign;
            dfab = Scalar(0.0);
            }

        Scalar fg = dab.x * dcbm.x + dab.y * dcbm.y + dab.z * dcbm.z;
        Scalar hg = ddc.x * dcbm.x + ddc.y * dcbm.y + ddc.z * dcbm.z;

        Scalar fga = fg * raa2inv * rginv;
        Scalar hgb = hg * rbb2inv * rginv;
        Scalar gaa = -raa2inv * rg;
        Scalar gbb = rbb2inv * rg;

        Scalar dtfx = gaa * aax;
        Scalar dtfy = gaa * aay;
        Scalar dtfz = gaa * aaz;
        Scalar dtgx = fga * aax - hgb * bbx;
        Scalar dtgy = fga * aay - hgb * bby;
        Scalar dtgz = fga * aaz - hgb * bbz;
        Scalar dthx = gbb * bbx;
        Scalar dthy = gbb * bby;
        Scalar dthz = gbb * bbz;

        //      Scalar df = -m_K[dihedral.type] * dfab;
        Scalar df
            = -m_K[dihedral_type] * dfab * Scalar(0.500); // the 0.5 term is for 1/2K in the forces

        Scalar sx2 = df * dtgx;
        Scalar sy2 = df * dtgy;
        Scalar sz2 = df * dtgz;

        Scalar ffax = df * dtfx;
        Scalar ffay = df * dtfy;
        Scalar ffaz = df * dtfz;

        Scalar ffbx = sx2 - ffax;
        Scalar ffby = sy2 - ffay;
        Scalar ffbz = sz2 - ffaz;

        Scalar ffdx = df * dthx;
        Scalar ffdy = df * dthy;
        Scalar ffdz = df * dthz;

        Scalar ffcx = -sx2 - ffdx;
        Scalar ffcy = -sy2 - ffdy;
        Scalar ffcz = -sz2 - ffdz;

        // Now, apply the force to each individual atom a,b,c,d
        // and accumulate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        // Scalar dihedral_eng = p*m_K[dihedral.type]*Scalar(1.0/4.0);
        Scalar dihedral_eng
            = p * m_K[dihedral_type] * Scalar(0.125); // the .125 term is (1/2)K * 1/4


        //smooth force and potential

        // compute squared cosines of angles
        Scalar r_12sq = dab.x*dab.x + dab.y*dab.y + dab.z*dab.z;
        Scalar r_34sq = ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z;
        Scalar r_23sq = dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z;

        // fg and hg have a different sign definition I think, which should not matter
        // due to being squared
        Scalar cos2_123 = fg*fg / (r_12sq * r_23sq);
        Scalar cos2_234 = hg*hg / (r_23sq * r_34sq);

        // evaluate the smoothing function and its derivative

        //todo-this should be defined outside of the computeForces for efficiency
        Scalar m_cos2_cut = fast::pow(fast::cos(m_cos_cut[dihedral_type]*M_PI/180.),2);
        Scalar m_cos2_on  = fast::pow(fast::cos(m_cos_cut[dihedral_type]*M_PI/180.),2);

        Scalar m_prefactor = 1. / fast::pow((m_cos2_cut - m_cos2_on),(m_m[dihedral_type]+1.));

        // symmetric smoothing is assumed around angle 90*, so no extra square root
        // calls are needed
        Scalar s_123, s_234;
        Scalar ds_123_divcos , ds_234_divcos;


        if (cos2_123 <= m_cos2_on)
        {
            s_123 = 1;
            ds_123_divcos = 0;
        } else if (cos2_123 >= m_cos2_cut)
        {
            s_123 = 0;
            ds_123_divcos = 0;
        }
        else
        {
            Scalar v = fast::pow(m_cos2_cut - cos2_123, m_m[dihedral_type]-1.);
            s_123 = m_prefactor * v * (m_cos2_cut - cos2_123) * (m_cos2_cut
                + m_m[dihedral_type] * cos2_123 - (m_m[dihedral_type]+1.) * m_cos2_on);
            ds_123_divcos = (m_prefactor * 2. * m_m[dihedral_type] * (m_m[dihedral_type]+1.) * v
                        * (m_cos2_on - cos2_123));

        }

        if (cos2_234 <= m_cos2_on)
        {
            s_234 = 1;
            ds_234_divcos = 0;
        } else if (cos2_234 >= m_cos2_cut)
        {
            s_234 = 0;
            ds_234_divcos = 0;
        }
        else
        {
            Scalar v = fast::pow(m_cos2_cut - cos2_234, m_m[dihedral_type]-1.);
            s_234 = m_prefactor * v * (m_cos2_cut - cos2_234) * (m_cos2_cut
                + m_m[dihedral_type] * cos2_234 - (m_m[dihedral_type]+1.) * m_cos2_on);
            ds_234_divcos = (m_prefactor * 2. * m_m[dihedral_type] * (m_m[dihedral_type]+1.) * v
                        * (m_cos2_on - cos2_234));

        }

        // compute gradients of the product s(123) s(234)
        Scalar3  grad_s_1 = make_scalar3(0,0,0);
        Scalar3  grad_s_2 = make_scalar3(0,0,0);
        Scalar3  grad_s_3 = make_scalar3(0,0,0);
        Scalar3  grad_s_4 = make_scalar3(0,0,0);

        // only need to do the calculation for 123 if s(234) has not already
        // switched off, and if the derivative of s(123) is nonzero
        if ( s_234 != 0. and ds_123_divcos != 0.)
        {
                // fg = r12_dot_r23
                // r_12 = dab r_23 = dcbm
                Scalar c_123 = s_234 * ds_123_divcos * fg/(r_12sq * r_23sq);
                Scalar3 grad_s123_1 = -c_123 * (dcbm - fg/r_12sq * dab);
                Scalar3 grad_s123_3 = c_123 * (dab - fg/r_23sq * dcbm);
                Scalar3 grad_s123_2 = -(grad_s123_1 + grad_s123_3);

                //add in contribution to the gradients
                grad_s_1 += grad_s123_1;
                grad_s_2 += grad_s123_2;
                grad_s_3 += grad_s123_3;
        }
        // only need to do the calculation for 234 if s(123) has not already
        // switched off, and if the derivative of s(234) is nonzero
        if( s_123 != 0. and ds_234_divcos != 0.)
        {
                // hg = r23_dot_r34
                // r_23 = dcbm
                // r_34 = -ddc
                Scalar3 r_34 = ddc*-1;
                Scalar c_234 = s_123 * ds_234_divcos * hg/(r_23sq * r_34sq);
                Scalar3 grad_s234_2 = -c_234 * (r_34 - hg/r_23sq * dcbm);
                Scalar3 grad_s234_4 = c_234 * (dcbm - hg/r_34sq * r_34);
                Scalar3 grad_s234_3 = -(grad_s234_2 + grad_s234_4);

                //add in contribution to the gradients
                grad_s_2 += grad_s234_2;
                grad_s_3 += grad_s234_3;
                grad_s_4 += grad_s234_4;
        }

        // apply the force smoothing
        Scalar s_prod = s_123 * s_234;
        Scalar dV_smooth = dihedral_eng - m_VS[dihedral_type];

        //F_1 = F_1 * s_prod - dV_smooth * grad_s_1;
        ffax = ffax * s_prod - dV_smooth * grad_s_1.x;
        ffay = ffay * s_prod - dV_smooth * grad_s_1.y;
        ffaz = ffaz * s_prod - dV_smooth * grad_s_1.z;


        //F_2 = F_2 * s_prod - dV_smooth * grad_s_2;
        ffbx = ffbx * s_prod - dV_smooth * grad_s_2.x;
        ffby = ffby * s_prod - dV_smooth * grad_s_2.y;
        ffbz = ffbz * s_prod - dV_smooth * grad_s_2.z;

        //F_3 = F_3 * s_prod - dV_smooth * grad_s_3;
        ffcx = ffcx * s_prod - dV_smooth * grad_s_3.x;
        ffcy = ffcy * s_prod - dV_smooth * grad_s_3.y;
        ffcz = ffcz * s_prod - dV_smooth * grad_s_3.z;

        //F_4 = F_4 * s_prod - dV_smooth * grad_s_4;
        ffdx = ffdx * s_prod - dV_smooth * grad_s_4.x;
        ffdy = ffdy * s_prod - dV_smooth * grad_s_4.y;
        ffdz = ffdz * s_prod - dV_smooth * grad_s_4.z;

        // apply the potential smoothing
        dihedral_eng = dV_smooth * s_prod + m_VS[dihedral_type];

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = (1. / 4.) * (dab.x * ffax + dcb.x * ffcx + (ddc.x + dcb.x) * ffdx);
        dihedral_virial[1] = (1. / 4.) * (dab.y * ffax + dcb.y * ffcx + (ddc.y + dcb.y) * ffdx);
        dihedral_virial[2] = (1. / 4.) * (dab.z * ffax + dcb.z * ffcx + (ddc.z + dcb.z) * ffdx);
        dihedral_virial[3] = (1. / 4.) * (dab.y * ffay + dcb.y * ffcy + (ddc.y + dcb.y) * ffdy);
        dihedral_virial[4] = (1. / 4.) * (dab.z * ffay + dcb.z * ffcy + (ddc.z + dcb.z) * ffdy);
        dihedral_virial[5] = (1. / 4.) * (dab.z * ffaz + dcb.z * ffcz + (ddc.z + dcb.z) * ffdz);

        h_force.data[idx_a].x += ffax;
        h_force.data[idx_a].y += ffay;
        h_force.data[idx_a].z += ffaz;
        h_force.data[idx_a].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_a] += dihedral_virial[k];

        h_force.data[idx_b].x += ffbx;
        h_force.data[idx_b].y += ffby;
        h_force.data[idx_b].z += ffbz;
        h_force.data[idx_b].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_b] += dihedral_virial[k];

        h_force.data[idx_c].x += ffcx;
        h_force.data[idx_c].y += ffcy;
        h_force.data[idx_c].z += ffcz;
        h_force.data[idx_c].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_c] += dihedral_virial[k];

        h_force.data[idx_d].x += ffdx;
        h_force.data[idx_d].y += ffdy;
        h_force.data[idx_d].z += ffdz;
        h_force.data[idx_d].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_d] += dihedral_virial[k];
        }
    }

namespace detail
    {
void export_SmoothedHarmonicDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<SmoothedHarmonicDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<SmoothedHarmonicDihedralForceCompute>>(m,
                                                                    "SmoothedHarmonicDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &SmoothedHarmonicDihedralForceCompute::setParamsPython)
        .def("getParams", &SmoothedHarmonicDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

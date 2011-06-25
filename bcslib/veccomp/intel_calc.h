/*
 * @file intel_calc.h
 *
 * Implementation of Calcuation functions using Intel IPP and MKL
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_INTEL_CALC_H_
#define BCSLIB_INTEL_CALC_H_

#include <bcslib/base/basic_defs.h>

#ifdef BCS_ENABLE_INTEL_IPPS
#include <bcslib/extern/ipps_select.h>
#define BCS_IPPS_CALL( statement ) if (n > 0) { ::statement; }
#endif

namespace bcs
{

#ifdef BCS_ENABLE_INTEL_IPPS

	/********************************************
	 *
	 *  Comparison
	 *
	 *******************************************/

	// max_each

	inline void vec_max_each(size_t n, const double* x1, const double* x2, double *y)
	{
		BCS_IPPS_CALL( ippsMaxEvery_64f(x1, x2, y, (Ipp32u)n) )
	}

	inline void vec_max_each(size_t n, const float* x1, const float* x2, float *y)
	{
		BCS_IPPS_CALL( ippsMaxEvery_32f(x1, x2, y, (Ipp32u)n) )
	}

	// min_each

	inline void vec_min_each(size_t n, const double* x1, const double* x2, double *y)
	{
		BCS_IPPS_CALL( ippsMinEvery_64f(x1, x2, y, (Ipp32u)n) )
	}

	inline void vec_min_each(size_t n, const float* x1, const float* x2, float *y)
	{
		BCS_IPPS_CALL( ippsMinEvery_32f(x1, x2, y, (Ipp32u)n) )
	}


	/********************************************
	 *
	 *  Bounding (Thresholding)
	 *
	 *******************************************/

	// lbound

	inline void vec_lbound(size_t n, const double* x, const double& lb, double *y)
	{
		BCS_IPPS_CALL( ippsThreshold_LT_64f(x, y, (int)n, lb) )
	}

	inline void vec_lbound(size_t n, const float* x, const float& lb, float *y)
	{
		BCS_IPPS_CALL( ippsThreshold_LT_32f(x, y, (int)n, lb) )
	}

	inline void vec_lbound_inplace(size_t n, double* y, const double& lb)
	{
		BCS_IPPS_CALL( ippsThreshold_LT_64f_I(y, (int)n, lb) )
	}

	inline void vec_lbound_inplace(size_t n, float* y, const float& lb)
	{
		BCS_IPPS_CALL( ippsThreshold_LT_32f_I(y, (int)n, lb) )
	}

	// ubound

	inline void vec_ubound(size_t n, const double* x, const double& ub, double *y)
	{
		BCS_IPPS_CALL( ippsThreshold_GT_64f(x, y, (int)n, ub) )
	}

	inline void vec_ubound(size_t n, const float* x, const float& ub, float *y)
	{
		BCS_IPPS_CALL( ippsThreshold_GT_32f(x, y, (int)n, ub) )
	}

	inline void vec_ubound_inplace(size_t n, double* y, const double& ub)
	{
		BCS_IPPS_CALL( ippsThreshold_GT_64f_I(y, (int)n, ub) )
	}

	inline void vec_ubound_inplace(size_t n, float* y, const float& ub)
	{
		BCS_IPPS_CALL( ippsThreshold_GT_32f_I(y, (int)n, ub) )
	}

	// abound

	inline void vec_abound(size_t n, const double* x, const double& ab, double *y)
	{
		BCS_IPPS_CALL( ippsThreshold_GTAbs_64f(x, y, (int)n, ab) )
	}

	inline void vec_abound(size_t n, const float* x, const float& ab, float *y)
	{
		BCS_IPPS_CALL( ippsThreshold_GTAbs_32f(x, y, (int)n, ab) )
	}

	inline void vec_abound_inplace(size_t n, double *y, const double& ab)
	{
		BCS_IPPS_CALL( ippsThreshold_GTAbs_64f_I(y, (int)n, ab) )
	}

	inline void vec_abound_inplace(size_t n, float *y, const float& ab)
	{
		BCS_IPPS_CALL( ippsThreshold_GTAbs_32f_I(y, (int)n, ab) )
	}



	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// addition

	inline void vec_add(size_t n, const double *x1, const double *x2, double *y)
	{
		BCS_IPPS_CALL( ippsAdd_64f(x1, x2, y, (int)n) )
	}

	inline void vec_add(size_t n, const float *x1, const float *x2, float *y)
	{
		BCS_IPPS_CALL( ippsAdd_32f(x1, x2, y, (int)n) )
	}

	inline void vec_add(size_t n, const double* x1, const double& x2, double *y)
	{
		BCS_IPPS_CALL( ippsAddC_64f(x1, x2, y, (int)n) )
	}

	inline void vec_add(size_t n, const float* x1, const float& x2, float *y)
	{
		BCS_IPPS_CALL( ippsAddC_32f(x1, x2, y, (int)n) )
	}

	inline void vec_add_inplace(size_t n, double *y, const double *x)
	{
		BCS_IPPS_CALL( ippsAdd_64f_I(x, y, (int)n) )
	}

	inline void vec_add_inplace(size_t n, float *y, const float *x)
	{
		BCS_IPPS_CALL( ippsAdd_32f_I(x, y, (int)n) )
	}

	inline void vec_add_inplace(size_t n, double *y, const double& x)
	{
		BCS_IPPS_CALL( ippsAddC_64f_I(x, y, (int)n) )
	}

	inline void vec_add_inplace(size_t n, float *y, const float& x)
	{
		BCS_IPPS_CALL( ippsAddC_32f_I(x, y, (int)n) )
	}


	// subtraction

	inline void vec_sub(size_t n, const double *x1, const double *x2, double *y)
	{
		BCS_IPPS_CALL( ippsSub_64f(x2, x1, y, (int)n) )
	}

	inline void vec_sub(size_t n, const float *x1, const float *x2, float *y)
	{
		BCS_IPPS_CALL( ippsSub_32f(x2, x1, y, (int)n) )
	}

	inline void vec_sub(size_t n, const double* x1, const double& x2, double *y)
	{
		BCS_IPPS_CALL( ippsSubC_64f(x1, x2, y, (int)n) )
	}

	inline void vec_sub(size_t n, const float* x1, const float& x2, float *y)
	{
		BCS_IPPS_CALL( ippsSubC_32f(x1, x2, y, (int)n) )
	}

	inline void vec_sub(size_t n, const double& x1, const double* x2, double *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_64f(x2, x1, y, (int)n) )
	}

	inline void vec_sub(size_t n, const float& x1, const float* x2, float *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_32f(x2, x1, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, double *y, const double *x)
	{
		BCS_IPPS_CALL( ippsSub_64f_I(x, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, float *y, const float *x)
	{
		BCS_IPPS_CALL( ippsSub_32f_I(x, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, double *y, const double& x)
	{
		BCS_IPPS_CALL( ippsSubC_64f_I(x, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, float *y, const float& x)
	{
		BCS_IPPS_CALL( ippsSubC_32f_I(x, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, const double& x, double *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_64f_I(x, y, (int)n) )
	}

	inline void vec_sub_inplace(size_t n, const float& x, float *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_32f_I(x, y, (int)n) )
	}


	// multiplication

	inline void vec_mul(size_t n, const double *x1, const double *x2, double *y)
	{
		BCS_IPPS_CALL( ippsMul_64f(x1, x2, y, (int)n) )
	}

	inline void vec_mul(size_t n, const float *x1, const float *x2, float *y)
	{
		BCS_IPPS_CALL( ippsMul_32f(x1, x2, y, (int)n) )
	}

	inline void vec_mul(size_t n, const double* x1, const double& x2, double *y)
	{
		BCS_IPPS_CALL( ippsMulC_64f(x1, x2, y, (int)n) )
	}

	inline void vec_mul(size_t n, const float* x1, const float& x2, float *y)
	{
		BCS_IPPS_CALL( ippsMulC_32f(x1, x2, y, (int)n) )
	}

	inline void vec_mul_inplace(size_t n, double *y, const double *x)
	{
		BCS_IPPS_CALL( ippsMul_64f_I(x, y, (int)n) )
	}

	inline void vec_mul_inplace(size_t n, float *y, const float *x)
	{
		BCS_IPPS_CALL( ippsMul_32f_I(x, y, (int)n) )
	}

	inline void vec_mul_inplace(size_t n, double *y, const double& x)
	{
		BCS_IPPS_CALL( ippsMulC_64f_I(x, y, (int)n) )
	}

	inline void vec_mul_inplace(size_t n, float *y, const float& x)
	{
		BCS_IPPS_CALL( ippsMulC_32f_I(x, y, (int)n) )
	}


	// division

	inline void vec_div(size_t n, const double *x1, const double *x2, double *y)
	{
		BCS_IPPS_CALL( ippsDiv_64f(x2, x1, y, (int)n) )
	}

	inline void vec_div(size_t n, const float *x1, const float *x2, float *y)
	{
		BCS_IPPS_CALL( ippsDiv_32f(x2, x1, y, (int)n) )
	}

	inline void vec_div(size_t n, const double* x1, const double& x2, double *y)
	{
		BCS_IPPS_CALL( ippsDivC_64f(x1, x2, y, (int)n) )
	}

	inline void vec_div(size_t n, const float* x1, const float& x2, float *y)
	{
		BCS_IPPS_CALL( ippsDivC_32f(x1, x2, y, (int)n) )
	}

	inline void vec_div_inplace(size_t n, double *y, const double *x)
	{
		BCS_IPPS_CALL( ippsDiv_64f_I(x, y, (int)n) )
	}

	inline void vec_div_inplace(size_t n, float *y, const float *x)
	{
		BCS_IPPS_CALL( ippsDiv_32f_I(x, y, (int)n) )
	}

	inline void vec_div_inplace(size_t n, double *y, const double& x)
	{
		BCS_IPPS_CALL( ippsDivC_64f_I(x, y, (int)n) )
	}

	inline void vec_div_inplace(size_t n, float *y, const float& x)
	{
		BCS_IPPS_CALL( ippsDivC_32f_I(x, y, (int)n) )
	}


	// negate

	inline void vec_negate(size_t n, const double *x, double *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_64f(x, 0.0, y, (int)n));
	}

	inline void vec_negate(size_t n, const float *x, float *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_32f(x, 0.0f, y, (int)n));
	}

	inline void vec_negate(size_t n, double *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_64f(y, 0.0, y, (int)n));
	}

	inline void vec_negate(size_t n, float *y)
	{
		BCS_IPPS_CALL( ippsSubCRev_32f(y, 0.0f, y, (int)n));
	}


	// absolute value

	inline void vec_abs(size_t n, const double *x, double *y)
	{
		BCS_IPPS_CALL( ippsAbs_64f(x, y, (int)n) )
	}

	inline void vec_abs(size_t n, const float *x, float *y)
	{
		BCS_IPPS_CALL( ippsAbs_32f(x, y, (int)n) )
	}



#endif // BCS_ENABLE_INTEL_IPPS

}


#endif

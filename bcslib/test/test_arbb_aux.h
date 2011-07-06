/**
 * @file test_arbb_aux.h
 *
 * Auxiliary facilities for testing ArBB functions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_TEST_ARBB_AUX_H_
#define BCSLIB_TEST_ARBB_AUX_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/test/test_assertion.h>

#include <arbb.hpp>
#include <cmath>

#define BCS_DEFAULT_ARBB_APPROX_TOLD 1e-12
#define BCS_DEFAULT_ARBB_APPROX_TOLF 1e-6f

namespace bcs
{
	namespace test
	{

		template<typename T>
		bool test_scalar_equal(const T& x, const typename arbb::uncaptured<T>::type& v)
		{
			return arbb::value(x) == v;
		}

		bool test_scalar_approx(const arbb::f64& x, const double& v, double tol = BCS_DEFAULT_ARBB_APPROX_TOLD)
		{
			return std::abs(arbb::value(x) - v) <= tol;
		}

		bool test_scalar_approx(const arbb::f32& x, const float& v, float tol = BCS_DEFAULT_ARBB_APPROX_TOLF)
		{
			return std::abs(arbb::value(x) - v) <= tol;
		}

		template<typename T>
		bool test_dense_size(const arbb::dense<T, 1>& a, size_t n)
		{
			return arbb::value(a.length()) == n;
		}

		template<typename T>
		bool test_dense_size(const arbb::dense<T, 2>& a, size_t nrows, size_t ncols)
		{
			return arbb::value(a.num_rows()) == nrows && arbb::value(a.num_cols()) == ncols &&
					arbb::value(a.length()) == nrows * ncols;
		}

		template<typename T, size_t D>
		bool test_dense_equal(const arbb::dense<T, D>& a, const T *r)
		{
			size_t n = arbb::value(a.length());
			arbb::const_range<T> rgn = a.read_only_range();
			for (size_t i = 0; i < n; ++i)
			{
				if (rgn[i] != r[i]) return false;
			}
			return true;
		}

		template<size_t D>
		bool test_dense_approx(const arbb::dense<arbb::f64, D>& a, const double *r, double tol = BCS_DEFAULT_ARBB_APPROX_TOLD)
		{
			size_t n = arbb::value(a.length());
			arbb::const_range<arbb::f64> rgn = a.read_only_range();
			for (size_t i = 0; i < n; ++i)
			{
				if (std::abs(rgn[i] - r[i]) > tol) return false;
			}
			return true;
		}

		template<size_t D>
		bool test_dense_approx(const arbb::dense<arbb::f32, D>& a, const float *r, float tol = BCS_DEFAULT_ARBB_APPROX_TOLF)
		{
			size_t n = arbb::value(a.length());
			arbb::const_range<arbb::f32> rgn = a.read_only_range();
			for (size_t i = 0; i < n; ++i)
			{
				if (std::abs(rgn[i] - r[i]) > tol) return false;
			}
			return true;
		}

	}
}


#endif


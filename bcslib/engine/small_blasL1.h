/**
 * @file small_blasL1.h
 *
 * Small size matrix BLAS Level 1
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_SMALL_BLASL1_H_
#define BCSLIB_SMALL_BLASL1_H_

#include <bcslib/engine/blas_kernel.h>
#include <cmath>

namespace bcs { namespace engine {


	/********************************************
	 *
	 *  small asum
	 *
	 ********************************************/

	template<typename T, int N>
	struct small_asum
	{
		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += std::fabs(x[i]);
			return s;
		};

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const int incx)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += std::fabs(x[i * incx]);
			return s;
		};

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			return incx == 1 ? eval_(x) : eval_(x, incx);
		}
	};

	template<typename T>
	struct small_asum<T, 1>
	{
		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x)
		{
			return std::fabs(x[0]);
		};

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const int incx)
		{
			return std::fabs(x[0]);
		};

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			return std::fabs(x[0]);
		}
	};


	/********************************************
	 *
	 *  small dot
	 *
	 ********************************************/

	template<typename T, int N>
	struct small_dot
	{
		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const T* __restrict__ y)
		{
			return dot_ker<T, N>::eval(x, y);
		}

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const int incx, const T* __restrict__ y)
		{
			return dot_ker<T, N>::eval(x, incx, y);
		}

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const T* __restrict__ y, const int incy)
		{
			return dot_ker<T, N>::eval(x, y, incy);
		}

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy)
		{
			return dot_ker<T, N>::eval(x, incx, y, incy);
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy)
		{
			if (N == 1)
			{
				return x[0] * y[0];
			}
			else
			{
				if (incx == 1)
				{
					return incy == 1 ? eval_(x, y) : eval_(x, y, incy);
				}
				else
				{
					return incy == 1 ? eval_(x, incx, y) : eval_(x, incx, y, incy);
				}
			}
		}
	};


	/********************************************
	 *
	 *  small nrm2
	 *
	 ********************************************/

	template<typename T, int N>
	struct small_nrm2
	{
		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x)
		{
			return N == 1 ? std::fabs(x[0]) : std::sqrt(vsqsum_ker<T, N>::eval(x));
		}

		BCS_ENSURE_INLINE
		static T eval_(const T* __restrict__ x, const int incx)
		{
			return N == 1 ? std::fabs(x[0]) : std::sqrt(vsqsum_ker<T, N>::eval(x, incx));
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			if (N == 1)
			{
				return std::fabs(x[0]);
			}
			else
			{
				return incx == 1 ? eval_(x) : eval_(x, incx);
			}
		}
	};


	/********************************************
	 *
	 *  small axpy
	 *
	 ********************************************/

	template<typename T, int N>
	struct small_axpy
	{
		BCS_ENSURE_INLINE
		static void eval_(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			if (a == 1)
				add_ker<T, N>::eval(x, y);
			else
				addx_ker<T, N>::eval(a, x, y);
		}


		BCS_ENSURE_INLINE
		static void eval_(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (a == 1)
				add_ker<T, N>::eval(x, incx, y);
			else
				addx_ker<T, N>::eval(a, x, incx, y);
		}


		BCS_ENSURE_INLINE
		static void eval_(const T a, const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (a == 1)
				add_ker<T, N>::eval(x, y, incy);
			else
				addx_ker<T, N>::eval(a, x, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (a == 1)
				add_ker<T, N>::eval(x, incx, y, incy);
			else
				addx_ker<T, N>::eval(a, x, incx, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (N == 1)
			{
				y[0] += a * x[0];
			}
			else
			{
				if (incx == 1)
				{
					if (incy == 1)
						eval_(a, x, y);
					else
						eval_(a, x, y, incy);
				}
				else
				{
					if (incy == 1)
						eval_(a, x, incx, y);
					else
						eval_(a, x, incx, y, incy);
				}
			}
		}

	};


	/********************************************
	 *
	 *  small axpy
	 *
	 ********************************************/

	template<typename T>
	BCS_ENSURE_INLINE void rot_pair(const T& c, const T& s, T& x, T& y)
	{
		T tx = x;
		x = c * tx + s * y;
		y = c * y - s * tx;
	}


	template<typename T, int N>
	struct small_rot
	{
		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) rot_pair(c, s, x[i], y[i]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) rot_pair(c, s, x[i * incx], y[i]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) rot_pair(c, s, x[i], y[i * incy]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) rot_pair(c, s, x[i * incx], y[i * incy]);
		}

		BCS_ENSURE_INLINE
		static void eval(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incx == 1)
			{
				if (incy == 1)
					eval_(c, s, x, y);
				else
					eval_(c, s, x, y, incy);
			}
			else
			{
				if (incy == 1)
					eval_(c, s, x, incx, y);
				else
					eval_(c, s, x, incx, y, incy);
			}
		}
	};

	template<typename T>
	struct small_rot<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, T* __restrict__ y)
		{
			rot_pair(c, s, x[0], y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			rot_pair(c, s, x[0], y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			rot_pair(c, s, x[0], y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval_(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			rot_pair(c, s, x[0], y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval(const T c, const T s, T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			rot_pair(c, s, x[0], y[0]);
		}
	};

} }

#endif /* SMALL_BLASL1_H_ */

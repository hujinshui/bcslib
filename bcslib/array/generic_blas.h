/**
 * @file array_blas.h
 *
 * Generic BLAS functions on 1D and 2D arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_BLAS_H_
#define BCSLIB_ARRAY_BLAS_H_

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/blas_base.h>

#include <type_traits>

namespace bcs
{
	namespace blas
	{

		// BLAS Level 1

		// asum

		template<class Arr>
		inline typename std::enable_if<is_array_view<Arr>::value,
		typename array_view_traits<Arr>::value_type>::type
		asum(const Arr& x)
		{
			typedef typename array_view_traits<Arr>::value_type value_type;
			BCS_STATIC_ASSERT_V(std::is_floating_point<value_type>);

			scoped_aview_read_proxy<Arr> xp(x);
			return _asum(make_cvec((int)get_num_elems(x), xp.pbase()));
		}

		// axpy

		template<class ArrX, class ArrY>
		inline typename std::enable_if<is_compatible_aviews<ArrX, ArrY>::value,
		void>::type
		axpy(const typename array_view_traits<ArrY>::value_type& alpha, const ArrX& x, ArrY& y)
		{
			typedef typename array_view_traits<ArrY>::value_type value_type;
			BCS_STATIC_ASSERT_V(std::is_floating_point<value_type>);

			check_arg(is_dense_view(y), "blas::axpy: y must be a dense view.");

			scoped_aview_read_proxy<ArrX> xp(x);
			return _axpy(alpha,
					make_cvec((int)get_num_elems(x), xp.pbase()),
					make_vec((int)get_num_elems(y), ptr_base(y)));
		}

		// dot

		template<class ArrX, class ArrY>
		inline typename std::enable_if<is_compatible_aviews<ArrX, ArrY>::value,
		typename array_view_traits<ArrX>::value_type>::type
		dot(const ArrX& x, const ArrY& y)
		{
			typedef typename array_view_traits<ArrX>::value_type value_type;
			BCS_STATIC_ASSERT_V(std::is_floating_point<value_type>);

			scoped_aview_read_proxy<ArrX> xp(x);
			scoped_aview_read_proxy<ArrY> yp(y);

			return _dot(
					make_cvec((int)get_num_elems(x), xp.pbase()),
					make_cvec((int)get_num_elems(y), yp.pbase()));
		}

		// nrm2

		template<class Arr>
		inline typename std::enable_if<is_array_view<Arr>::value,
		typename array_view_traits<Arr>::value_type>::type
		nrm2(const Arr& x)
		{
			typedef typename array_view_traits<Arr>::value_type value_type;
			BCS_STATIC_ASSERT_V(std::is_floating_point<value_type>);

			scoped_aview_read_proxy<Arr> xp(x);
			return _nrm2(make_cvec((int)get_num_elems(x), xp.pbase()));
		}

		// rot

		template<class ArrX, class ArrY>
		inline typename std::enable_if<is_compatible_aviews<ArrX, ArrY>::value,
		void>::type
		rot(ArrX& x, ArrY& y,
				const typename array_view_traits<ArrY>::value_type& c,
				const typename array_view_traits<ArrY>::value_type& s)
		{
			typedef typename array_view_traits<ArrX>::value_type value_type;
			BCS_STATIC_ASSERT_V(std::is_floating_point<value_type>);

			check_arg(is_dense_view(x), "blas::rot: x must be a dense view.");
			check_arg(is_dense_view(y), "blas::rot: y must be a dense view.");

			return _rot(
					make_vec((int)get_num_elems(x), ptr_base(x)),
					make_vec((int)get_num_elems(y), ptr_base(y)), c, s);
		}



	}

}

#endif 


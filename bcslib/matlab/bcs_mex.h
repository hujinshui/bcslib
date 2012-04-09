/**
 * @file bcs_mex.h
 *
 * The interface between bcslib and matlab mex
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MEX_H
#define BCSLIB_MEX_H

#include <bcslib/matlab/matlab_base.h>
#include <bcslib/matlab/marray.h>

#include <bcslib/array/aview1d.h>
#include <bcslib/array/aview2d.h>
#include <bcslib/array/aview1d_ops.h>
#include <bcslib/array/aview2d_ops.h>

#include <cstring>
#include <vector>
#include <iterator>

namespace bcs {  namespace matlab {

	// view from marray

	template<typename T>
	inline caview1d<T> view1d(const_marray a)
	{
		return caview1d<T>(a.data<T>(), a.nelems());
	}

	template<typename T>
	inline aview1d<T> view1d(marray a)
	{
		return aview1d<T>(a.data<T>(), a.nelems());
	}

	template<typename T>
	inline caview2d<T> view2d(const_marray a)
	{
		return caview2d<T>(a.data<T>(), a.nrows(), a.ncolumns());
	}

	template<typename T>
	inline aview2d<T> view2d(marray a)
	{
		return aview2d<T>(a.data<T>(), a.nrows(), a.ncolumns());
	}


	// to matlab vector

	template<class Derived, typename T>
	inline marray to_matlab_row(const IConstRegularAView1D<Derived, T>& v)
	{
		marray a = create_marray<T>(1, v.nelems());
		aview1d<T> dst = view1d<T>(a);
		copy(v.derived(), dst);
		return a;
	}

	template<class Derived, typename T>
	inline marray to_matlab_column(const IConstRegularAView1D<Derived, T>& v)
	{
		marray a = create_marray<T>(v.nelems(), 1);
		aview1d<T> dst = view1d<T>(a);
		copy(v.derived(), dst);
		return a;
	}


	template<typename T>
	inline marray to_matlab_row(const std::vector<T>& v)
	{
		index_t n = (index_t)v.size();
		marray a = create_marray<T>(1, n);
		mem<T>::copy(size_t(n), &(v[0]), a.data<T>());
		return a;
	}

	template<typename T>
	inline marray to_matlab_column(const std::vector<T>& v)
	{
		index_t n = (index_t)v.size();
		marray a = create_marray<T>(n, 1);
		mem<T>::copy(size_t(n), &(v[0]), a.data<T>());
		return a;
	}


	template<typename TIter>
	inline marray to_matlab_row(TIter first, index_t n)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;

		marray a = create_marray<T>(1, n);
		T *pa = a.data<T>();
		for (index_t i = 0; i < n; ++i) *(pa++) = *(first++);
		return a;
	}

	template<typename TIter>
	inline marray to_matlab_column(TIter first, index_t n)
	{
		typedef typename std::iterator_traits<TIter>::value_type T;

		marray a = create_marray<T>(n, 1);
		T *pa = a.data<T>();
		for (index_t i = 0; i < n; ++i) *(pa++) = *(first++);
		return a;
	}

	// to matlab matrix

	template<class Derived, typename T>
	marray to_matlab_matrix(const IConstRegularAView2D<Derived, T>& v)
	{
		marray a = create_marray<T>(v.nrows(), v.ncolumns());
		aview2d<T> dst = view2d<T>(a);
		copy(v.derived(), dst);
		return a;
	}

} }


#define BCSMEX_MAINDEF \
		void bcsmex_main(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]); \
		void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) \
		{ \
			try { \
				bcsmex_main(nlhs, plhs, nrhs, prhs); \
			} \
			catch(bcs::matlab::mexception& mexc) { \
				mexErrMsgIdAndTxt(mexc.identifier(), mexc.message()); \
			} \
		}


#endif 

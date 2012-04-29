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

#include <bcslib/matrix.h>

#include <cstring>
#include <vector>
#include <iterator>

namespace bcs {  namespace matlab {

	// view from marray

	template<typename T>
	inline cref_row<T> as_row(const_marray a)
	{
		return cref_row<T>(a.data<T>(), a.nelems());
	}

	template<typename T>
	inline ref_row<T> as_row(marray a)
	{
		return ref_row<T>(a.data<T>(), a.nelems());
	}

	template<typename T>
	inline cref_col<T> as_column(const_marray a)
	{
		return cref_col<T>(a.data<T>(), a.nelems());
	}

	template<typename T>
	inline ref_col<T> as_column(marray a)
	{
		return ref_col<T>(a.data<T>(), a.nelems());
	}


	template<typename T>
	inline cref_matrix<T> as_mat(const_marray a)
	{
		return cref_matrix<T>(a.data<T>(), a.nrows(), a.ncolumns());
	}

	template<typename T>
	inline ref_matrix<T> view2d(marray a)
	{
		return ref_matrix<T>(a.data<T>(), a.nrows(), a.ncolumns());
	}


	// to matlab vector / matrix

	template<class Derived, typename T>
	inline marray to_matlab_row(const IRegularMatrix<Derived, T>& mat)
	{
		marray a = create_marray<T>(1, mat.nelems());
		ref_row<T> dst = as_row<T>(a);
		copy(mat.derived(), dst);
		return a;
	}

	template<class Derived, typename T>
	inline marray to_matlab_column(const IRegularMatrix<Derived, T>& mat)
	{
		marray a = create_marray<T>(mat.nelems(), 1);
		ref_col<T> dst = as_column<T>(a);
		copy(mat.derived(), dst);
		return a;
	}

	template<class Derived, typename T>
	marray to_matlab_matrix(const IRegularMatrix<Derived, T>& mat)
	{
		marray a = create_marray<T>(mat.nrows(), mat.ncolumns());
		ref_matrix<T> dst = as_mat<T>(a);
		copy(mat.derived(), dst);
		return a;
	}

	template<typename T>
	inline marray to_matlab_row(const std::vector<T>& v)
	{
		index_t n = (index_t)v.size();
		marray a = create_marray<T>(1, n);
		copy_elems(n, &(v[0]), a.data<T>());
		return a;
	}

	template<typename T>
	inline marray to_matlab_column(const std::vector<T>& v)
	{
		index_t n = (index_t)v.size();
		marray a = create_marray<T>(n, 1);
		copy_elems(n, &(v[0]), a.data<T>());
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

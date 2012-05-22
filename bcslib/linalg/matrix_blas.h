/**
 * @file matrix_blas.h
 *
 * BLAS wrappers based on matrix interfaces
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_BLAS_H_
#define BCSLIB_MATRIX_BLAS_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/engine/blas.h>

namespace bcs { namespace blas {


	/********************************************
	 *
	 *  BLAS Level 1
	 *
	 ********************************************/

	// asum

	template<typename T, class Mat>
	inline typename enable_ty<
		is_floating_point<T>::value &&
		has_continuous_layout<Mat>::value,
	T>::type
	asum(const IDenseMatrix<Mat, T>& x)
	{
		return engine::asum<T, ct_size<Mat>::value>::eval(
				(int)x.nelems(), x.ptr_data());
	}

	// axpy

	template<typename T, class MatX, class MatY>
	inline typename enable_ty<
		is_floating_point<T>::value &&
		has_continuous_layout<MatX>::value &&
		has_continuous_layout<MatY>::value,
	void>::type
	axpy(const T a, const IDenseMatrix<MatX, T>& x, IDenseMatrix<MatY, T>& y)
	{
		engine::axpy<T, binary_ct_size<MatX, MatY>::value>::eval(
				(int)x.nelems(), a, x.ptr_data(), y.ptr_data());
	}

	// dot

	template<typename T, class MatX, class MatY>
	inline typename enable_ty<
		is_floating_point<T>::value &&
		has_continuous_layout<MatX>::value &&
		has_continuous_layout<MatY>::value,
	T>::type
	dot(const IDenseMatrix<MatX, T>& x, const IDenseMatrix<MatY, T>& y)
	{
		check_arg( has_same_size(x, y), "x and y should have the same size" );

		return engine::dot<T, binary_ct_size<MatX, MatY>::value>::eval(
				(int)x.nelems(), x.ptr_data(), y.ptr_data());
	}

	// nrm2

	template<typename T, class Mat>
	inline typename enable_ty<
		is_floating_point<T>::value &&
		has_continuous_layout<Mat>::value,
	T>::type
	nrm2(const IDenseMatrix<Mat, T>& x)
	{
		return engine::nrm2<T, ct_size<Mat>::value>::eval(
				(int)x.nelems(), x.ptr_data());
	}

	// rot

	template<typename T, class MatX, class MatY>
	inline typename enable_ty<
		is_floating_point<T>::value &&
		has_continuous_layout<MatX>::value &&
		has_continuous_layout<MatY>::value,
	void>::type
	rot(IDenseMatrix<MatX, T>& x, IDenseMatrix<MatY, T>& y, const T c, const T s)
	{
		engine::rot<T, binary_ct_size<MatX, MatY>::value>::eval(
				(int)x.nelems(), x.ptr_data(), y.ptr_data(), c, s);
	}


	/********************************************
	 *
	 *  BLAS Level 2
	 *
	 ********************************************/

	// gemv

	template<typename T, class MatA, class MatX, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemv_n(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatX, T>& x,
			const T beta, IDenseMatrix<MatY, T>& y)
	{
		check_arg(x.nrows() == a.ncolumns() && x.ncolumns() == 1,
				"The size of x is invalid (for gemv_n)");

		check_arg(y.nrows() == a.nrows() && y.ncolumns() == 1,
				"The size of y is invalid (for gemv_n)");

		typedef engine::gemv<T, ct_rows<MatA>::value, ct_cols<MatA>::value> impl_t;
		impl_t::eval('N', (int)a.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(),
				beta, y.ptr_data());
	}

	template<typename T, class MatA, class MatX, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemv_t(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatX, T>& x,
			const T beta, IDenseMatrix<MatY, T>& y)
	{
		check_arg(x.nrows() == a.nrows() && x.ncolumns() == 1,
				"The size of x is invalid (for gemv_t)");

		check_arg(y.nrows() == a.ncolumns() && y.ncolumns() == 1,
				"The size of y is invalid (for gemv_t)");

		typedef engine::gemv<T, ct_rows<MatA>::value, ct_cols<MatA>::value> impl_t;
		impl_t::eval('T', (int)a.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(),
				beta, y.ptr_data());
	}


	// gevm

	template<typename T, class MatX, class MatA, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gevm_n(const T alpha, const IDenseMatrix<MatX, T>& x, const IDenseMatrix<MatA, T>& a,
			const T beta, IDenseMatrix<MatY, T>& y)
	{
		check_arg(x.nrows() == 1 && x.ncolumns() == a.nrows(),
				"The size of x is invalid (for gevm_n)");

		check_arg(y.nrows() == 1 && y.ncolumns() == a.ncolumns(),
				"The size of y is invalid (for gevm_n)");

		typedef engine::gemv_ex<T, ct_rows<MatA>::value, ct_cols<MatA>::value> impl_t;

		impl_t::eval('T', (int)a.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(), (int)x.lead_dim(),
				beta, y.ptr_data(), (int)y.lead_dim());
	}

	template<typename T, class MatX, class MatA, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gevm_t(const T alpha, const IDenseMatrix<MatX, T>& x, const IDenseMatrix<MatA, T>& a,
			const T beta, IDenseMatrix<MatY, T>& y)
	{
		check_arg(x.nrows() == 1 && x.ncolumns() == a.ncolumns(),
				"The size of x is invalid (for gevm_t)");

		check_arg(y.nrows() == 1 && y.ncolumns() == a.nrows(),
				"The size of y is invalid (for gevm_t)");

		typedef engine::gemv_ex<T, ct_rows<MatA>::value, ct_cols<MatA>::value> impl_t;

		impl_t::eval('N', (int)a.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(), (int)x.lead_dim(),
				beta, y.ptr_data(), (int)y.lead_dim());
	}




	// ger

	template<typename T, class MatX, class MatY, class MatA>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	ger(const T alpha, const IDenseMatrix<MatX, T>& x, const IDenseMatrix<MatY, T>& y,
			IDenseMatrix<MatA, T>& a)
	{
		check_arg(x.nrows() == a.nrows() && x.ncolumns() == 1,
				"The size of x is invalid (for ger)");

		check_arg(y.nrows() == a.ncolumns() && y.ncolumns() == 1,
				"The size of y is invalid (for ger)");

		typedef engine::ger<T,
				binary_ctdim<ct_rows<MatA>::value, ct_rows<MatX>::value>::value,
				binary_ctdim<ct_cols<MatA>::value, ct_rows<MatY>::value>::value> impl_t;

		impl_t::eval((int)a.nrows(), (int)a.ncolumns(),
				alpha, x.ptr_data(), y.ptr_data(), a.ptr_data(), (int)a.lead_dim());
	}


	// symv

	template<typename T, class MatX, class MatA, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	symv(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatX, T>& x,
			const T beta, IDenseMatrix<MatY, T>& y, const char uplo = 'L')
	{
		const index_t n = a.nrows();

		check_arg(a.ncolumns() == n, "a should be a square matrix (for symv)");
		check_arg(x.nrows() == n && x.ncolumns() == 1, "The size of x is invalid (for symv)");

		typedef engine::symv<T,
				binary_ctdim<ct_rows<MatA>::value, ct_cols<MatA>::value>::value> impl_t;

		impl_t::eval(uplo, (int)n,
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(),
				beta, y.ptr_data());
	}


	// syvm

	template<typename T, class MatA, class MatX, class MatY>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	syvm(const T alpha, const IDenseMatrix<MatX, T>& x, const IDenseMatrix<MatA, T>& a,
			const T beta, IDenseMatrix<MatY, T>& y, const char uplo = 'L')
	{
		const index_t n = a.nrows();

		check_arg(a.ncolumns() == n, "a should be a square matrix (for symv)");
		check_arg(x.ncolumns() == n && x.nrows() == 1, "The size of x is invalid (for syvm)");

		typedef engine::symv_ex<T,
				binary_ctdim<ct_rows<MatA>::value, ct_cols<MatA>::value>::value> impl_t;

		impl_t::eval(uplo, (int)n,
				alpha, a.ptr_data(), (int)a.lead_dim(), x.ptr_data(), (int)x.lead_dim(),
				beta, y.ptr_data(), (int)y.lead_dim());
	}



	/********************************************
	 *
	 *  BLAS Level 3
	 *
	 ********************************************/

	// gemm

	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemm_nn(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c)
	{
		check_arg(a.ncolumns() == b.nrows(), "Inconsistent inner dimension for gemm_nn");
		check_arg(c.nrows() == a.nrows() && c.ncolumns() == b.ncolumns(), "The size of c is invalid for gemm_nn");

		typedef engine::gemm<T,
				binary_ctdim<ct_rows<MatA>::value, ct_rows<MatC>::value>::value,
				binary_ctdim<ct_cols<MatB>::value, ct_cols<MatC>::value>::value,
				binary_ctdim<ct_cols<MatA>::value, ct_rows<MatB>::value>::value> impl_t;

		impl_t::eval('N', 'N', (int)a.nrows(), (int)b.ncolumns(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}


	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemm_nt(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c)
	{
		check_arg(a.ncolumns() == b.ncolumns(), "Inconsistent inner dimension for gemm_nn");
		check_arg(c.nrows() == a.nrows() && c.ncolumns() == b.nrows(), "The size of c is invalid for gemm_nt");

		typedef engine::gemm<T,
				binary_ctdim<ct_rows<MatA>::value, ct_rows<MatC>::value>::value,
				binary_ctdim<ct_rows<MatB>::value, ct_cols<MatC>::value>::value,
				binary_ctdim<ct_cols<MatA>::value, ct_cols<MatB>::value>::value> impl_t;

		impl_t::eval('N', 'T', (int)a.nrows(), (int)b.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}

	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemm_tn(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c)
	{
		check_arg(a.nrows() == b.nrows(), "Inconsistent inner dimension for gemm_nn");
		check_arg(c.nrows() == a.ncolumns() && c.ncolumns() == b.ncolumns(), "The size of c is invalid for gemm_tn");

		typedef engine::gemm<T,
				binary_ctdim<ct_cols<MatA>::value, ct_rows<MatC>::value>::value,
				binary_ctdim<ct_cols<MatB>::value, ct_cols<MatC>::value>::value,
				binary_ctdim<ct_rows<MatA>::value, ct_rows<MatB>::value>::value> impl_t;

		impl_t::eval('T', 'N', (int)a.ncolumns(), (int)b.ncolumns(), (int)a.nrows(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}

	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	gemm_tt(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c)
	{
		check_arg(a.nrows() == b.ncolumns(), "Inconsistent inner dimension for gemm_nn");
		check_arg(c.nrows() == a.ncolumns() && c.ncolumns() == b.nrows(), "The size of c is invalid for gemm_tt");

		typedef engine::gemm<T,
				binary_ctdim<ct_cols<MatA>::value, ct_rows<MatC>::value>::value,
				binary_ctdim<ct_rows<MatB>::value, ct_cols<MatC>::value>::value,
				binary_ctdim<ct_rows<MatA>::value, ct_cols<MatB>::value>::value> impl_t;

		impl_t::eval('T', 'T', (int)a.ncolumns(), (int)b.nrows(), (int)a.nrows(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}

	// symm

	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	symm_l(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c, const char uplo = 'L')
	{
		check_arg(a.nrows() == a.ncolumns(), "a should be a square matrix for symm_l");
		check_arg(a.ncolumns() == b.nrows(), "Inconsistent inner dimension for symm_l");

		typedef engine::symm<T,
				binary_ctdim<ct_rows<MatA>::value, ct_cols<MatA>::value>::value,
				binary_ctdim<ct_cols<MatB>::value, ct_cols<MatC>::value>::value> impl_t;

		impl_t::eval('L', uplo, (int)a.nrows(), (int)b.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}

	template<typename T, class MatA, class MatB, class MatC>
	inline typename enable_ty<is_floating_point<T>::value, void>::type
	symm_r(const T alpha, const IDenseMatrix<MatA, T>& a, const IDenseMatrix<MatB, T>& b,
			const T beta, IDenseMatrix<MatC, T>& c, const char uplo = 'L')
	{
		check_arg(a.nrows() == a.ncolumns(), "a should be a square matrix for symm_r");
		check_arg(b.ncolumns() == a.nrows(), "Inconsistent inner dimension for symm_r");

		typedef engine::symm<T,
				binary_ctdim<ct_rows<MatB>::value, ct_rows<MatC>::value>::value,
				binary_ctdim<ct_rows<MatA>::value, ct_cols<MatA>::value>::value> impl_t;

		impl_t::eval('R', uplo, (int)b.nrows(), (int)a.ncolumns(),
				alpha, a.ptr_data(), (int)a.lead_dim(), b.ptr_data(), (int)b.lead_dim(),
				beta, c.ptr_data(), (int)c.lead_dim());
	}


} }

#endif

/**
 * @file cuda_timer.h
 *
 * The class to implement a timer based on CUDA runtime
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_CUDA_TIMER_H_
#define BCSLIB_CUDA_TIMER_H_

namespace bcs { namespace cuda {

	class event_timer
	{
	public:
	    event_timer(bool to_start = false)
		{
			cudaEventCreate(&m_start);
			cudaEventCreate(&m_stop);

			if (to_start) start();
		}

		void start()
		{
			cudaEventRecord(m_start, 0);
		}

		void stop()
		{
		    cudaEventRecord(m_stop, 0);
			cudaEventSynchronize(m_stop);
		}

		double elapsed_milliseconds() const
		{
			float ms;
			cudaEventElapsedTime(&ms, m_start, m_stop);
			return double(ms);
		}

		double elapsed_seconds() const
		{
			return elapsed_milliseconds() * 1.0e-3;
		}

	private:
		cudaEvent_t m_start;
		cudaEvent_t m_stop;

	}; // end event_timer

} }

#endif


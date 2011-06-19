/*
 * @file performance_timer.h
 *
 * A timer class to measure elapsed time
 *
 *  Created on: May 9, 2011
 *      Author: dhlin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_PERFORMANCE_TIMER_H
#define BCSLIB_PERFORMANCE_TIMER_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>

#include <ctime>

namespace bcs
{
	namespace test
	{


		/**
		 * A performance timer
		 *
		 * TODO: re-implement using high accuracy clock
		 */
		class performance_timer
		{
		public:
			explicit performance_timer(bool start_it = false)
			: m_is_on(false), m_last_resume(0), m_elapsed(0)
			{
				if (start_it)
				{
					start();
				}
			}

			void start()
			{
				m_elapsed = 0;
				resume();
			}


			void reset()
			{
				m_is_on = false;
				m_last_resume = 0;
				m_elapsed = 0;
			}

			void resume()
			{
				m_is_on = true;
				m_last_resume = std::clock();
			}

			void pause()
			{
				m_elapsed += last_elapsed();
				m_is_on = false;
			}

			double elapsed_seconds() const
			{
				std::clock_t e = m_is_on ? last_elapsed() + m_elapsed : m_elapsed;
				return double(e) / CLOCKS_PER_SEC;
			}


		private:
			std::clock_t last_elapsed() const
			{
				return std::clock() - m_last_resume;
			}

		private:
			bool m_is_on;
			std::clock_t m_last_resume;
			std::clock_t m_elapsed;

		}; // end class performance timer

	}
}

#endif

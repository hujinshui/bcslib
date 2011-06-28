#!/usr/bin/env python

# This is a simple script to generate makefile for different platforms
#
#	This script tries to generate a makefile as simple as possible
#	so as to avoid possible problems arising in auto-build tools
#

# Copyright (c) 2011 Dahua Lin, MIT

import platform
import os
import os.path
import sys
import subprocess as sp
import ConfigParser as cp
import re

# Auxiliary functions

def dequote(sval):
	"""Dequote a string value"""
	
	if len(sval) >= 2 and (sval[0]=='"' and sval[-1]=='"'):
		return sval[1:-1]
	else:
		return sval
	

def report_err(message):
	sys.stderr.write(message + '\n')
	sys.exit(1)


def get_cmd_output(cmd):
	"""Get the stdout and stderr output of a command"""
	
	p = sp.Popen(cmd, shell=True, 
		stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
		
	return p.communicate()
	
def found_sys_command(name):
	"""Test whether a particular command is available in search path"""
	
	rout, rerr = get_cmd_output('which ' + name)
	return len(rout) > 0
	
def ver2tuple(vstr):
	"""Convert version string to tuple"""
	
	vns = tuple(int(x) for x in vstr.strip().split('.'))
	if len(vns) > 3:
		vns = vns[:3]
	elif len(vns) == 2:
		vns = (vns[0], vns[1], 0)
	elif len(vns) == 1:
		vns = (vns[0], 0, 0)
		
	return vns
	
# compiler version retrieval	
	
def get_compiler_version(name):
	"""Get the tuple that represents the version of a compiler"""
	
	if name == 'g++':
		rout, rerr = get_cmd_output('g++ -dumpversion')
		if len(rout) > 0:
			return ver2tuple(rout)
		else:
			report_err('Failed to get the version of g++')
			
	elif name == 'clang++':
		rout, rerr = get_cmd_output('clang++ --version')
		if len(rout) > 0:
			terms = rout.strip().split()
			if len(terms) >= 3:
				return ver2tuple(terms[2])
				
		report_err('Failed to get the version of clang++')
			
	else:
		report_err('Unsupported compiler name ' + name)
			
			
# parse ini file

def test_platform(platform_req, platform_spec):
	"""test whether the input platform matches the requirement"""
	
	platform_req = dequote(platform_req)
	
	if platform_req == None:
		return True
	else:
		reqs = platform_req.split(';')
		for req in reqs:
			terms = req.strip().split('-')
			if len(terms) == 1:
				if terms[0].lower() == platform_spec[0].lower():
					return True
			elif len(terms) == 2:
				if terms[0].lower() == platform_spec[0].lower() and terms[1].lower() == platform_spec[1].lower():
					return True
			else:
				report_err("Invalid platform requirement string: %s" % req.strip())
				
		return False
			
			
def test_compiler(compiler_req, compiler_spec):
	"""test whether the input compiler matches the requirement"""
	
	compiler_req = dequote(compiler_req)
	
	if compiler_req == None:
		return True
	else:
		reqs = compiler_req.split(';')
		for req in reqs:
			terms = req.strip().split('-')
			if len(terms) == 1:
				if terms[0].lower() == compiler_spec[0].lower():
					return True
			elif len(terms) == 2:
				if terms[0].lower() == compiler_spec[0].lower():
					rver = ver2tuple(terms[1])
					if rver <= compiler_spec[1]:
						return True
			
		return False
			
	
def add_valid_entries(vdict, indict):
	"""Add valid entries from an input dict to vdict"""
	
	for name, val in indict.iteritems():
		if len(name) > 0 and name[0] != '_':
			vdict[name] = dequote(val.strip())
			
			
def parse_cfg(filename, platform_spec, compiler_spec):
	"""parse init file and make a variable context for makefile generation"""
	
	cfg = cp.ConfigParser()
	cfg.read(filename)
	
	# check basic requirement
	
	dsec = cfg.defaults()

	if '_platforms_' in dsec:
		if not test_platform(dsec['_platforms_'], platform_spec):
			report_err("The current system does not meet the platform requirement.")
	else:
		report_err("The option _platforms_ is missing from the default section.")
		
	if '_compilers_' in dsec:
		if not test_compiler(dsec['_compilers_'], compiler_spec):
			report_err("The current system does not meet the compiler requirement.")
	else:
		report_err("The option _compilers_ is missing from the default section.")
	
	# add matched sections
	
	print 'section DEFAULT included'
	vdict = dict()
	add_valid_entries(vdict, dsec)
	
	for sec in cfg.sections():
		has_platforms_req = cfg.has_option(sec, '_platforms_')
		has_compilers_req = cfg.has_option(sec, '_compilers_')
					
		platform_reqs = cfg.get(sec, '_platforms_') if has_platforms_req else None
		compiler_reqs = cfg.get(sec, '_compilers_') if has_compilers_req else None
		
		if test_platform(platform_reqs, platform_spec) and test_compiler(compiler_reqs, compiler_spec):
			
			# include all entries in this section (may override previous)
			print 'section %s included' % sec
			add_valid_entries(vdict, dict(cfg.items(sec)))
					
	return vdict
	
	
def generate_file(vdict, templatefile, outputfile):
	"""generate a file from a template file"""
	
	pat = re.compile('@\w[\w\d_]+@')
	
	# read from template
	with open(templatefile, 'r') as fin:
		lines = fin.readlines()
		
	i = 0
	for line in lines:
		i = i + 1
		
		sl = line.strip()
		if len(sl) == 0 or sl[0] == '#':  # empty line or comment line
			continue
		
		ms = re.findall(pat, line)
		if len(ms) > 0:
			for m in ms:
				vname = m[1:-1]
				if vname in vdict:
					vval = vdict[vname]
					line = line.replace(m, vval)
				else:
					print "WARNING: variable %s in line %d does not find a value" % (vname, i)
		
			lines[i-1] = line
		
	# write to destination
	with open(outputfile, 'w') as fout:
		for line in lines:
			fout.write(line)
	
	
			
# main procedure

if __name__ == '__main__':
	
	print "smake started ..."
	
	# detect platform
	
	os_name = platform.system()
	arch = platform.architecture()
	arch_bits = arch[0]
	platform_spec = (os_name, arch_bits)
	
	print "Detected platform: (%s, %s)" % platform_spec
	
	# detect compiler
	
	cxx = os.getenv('CXX')
	if cxx == None:
		if found_sys_command('g++'):
			cxx = 'g++'
		elif found_sys_command('clang++'):
			cxx = 'clang++'
		else:
			report_err('Unable to detect a supported compiler')
			
	# detect compiler version
	
	ver = get_compiler_version(cxx)
	vstr = '.'.join(str(x) for x in ver)
	
	compiler_spec = (cxx, ver)
	print "Detected compiler: (%s, %s)" % (cxx, vstr)
	
	# parse ini file
	
	cfgfile = 'smake.ini'
	if not os.path.isfile(cfgfile):
		report_err('The configuration file %s is not found.' % cfgfile)
		
	vdict = parse_cfg(cfgfile, platform_spec, compiler_spec)
	
	appm_count = 0
	if len(vdict) > 0:
		print "Applicable macros:"
		for name, val in vdict.items():
			if name[-1] != '_':
				appm_count = appm_count + 1
				print '    ', name, '=', val
	
	if appm_count == 0:
		print "No appliable macros"
		
	vdict['cxx'] = cxx
		
	# generate makefile
		
	infile = vdict['input_file_'] or 'makefile.in'
	outfile = vdict['output_file_'] or 'makefile'
	
	if not os.path.isfile(infile):
		report_err('The input template %s is not found.' % infile)
	
	generate_file(vdict, infile, outfile)
		
	print "%s is generated" % outfile	
	print "smake done!\n"


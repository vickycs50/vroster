import os
import sys
import traceback
import socket
from ctypes import *
import numpy 
from numpy.ctypeslib import ndpointer
import inspect
import string
import platform 



class Matlab: 
	
	class Local:
		def __init__(self, name='[]'):
			self.name = name

		def __str__(self):
			return self.name
	
	
	"""
	Loads Matlab libraries and engine. If Matlab engine fails to load an exception is raised.

	version: integer indicating the Matlab version (ex. 7)
	matlabLocation: full path to the root directory where Matlab is located
	libdir: Internal directory where all shared Matlab libraries are located (ex. glnx86)

	"""
	def __init__(self, version, matlabLocation, libdir):
		currentOS = os.uname()[0]
		suff = None
		if currentOS == 'Darwin':
			suff = 'dylib'
			self.c = cdll.LoadLibrary('libc.%s'%suff)
		elif os.name=='posix':
			suff = 'so'
			self.c = cdll.LoadLibrary('libc.%s.6'%suff)
		

		self.version = version
		ldir = '/bin/'+libdir+'/'
		try:
			self.eng = cdll.LoadLibrary('%s/%s/%s.%s' % (matlabLocation, ldir, 'libeng', suff))
			self.mat = cdll.LoadLibrary('%s/%s/%s.%s' % (matlabLocation, ldir, 'libmat', suff))
			self.mx = cdll.LoadLibrary('%s/%s/%s.%s' % (matlabLocation, ldir, 'libmx', suff))
			self.mex = cdll.LoadLibrary('%s/%s/%s.%s' % (matlabLocation, ldir, 'libmex', suff))
		except Exception as e:
			traceback.print_exc()
			#sys.exit()
		self.eng.engOpen.argtypes = [c_char_p]
		self.eng.engOpen.restype = c_void_p
		self.matlab = self.eng.engOpen(matlabLocation+'/bin/matlab -nosplash -nodisplay')

 		if self.matlab == None:
			raise 'Error opening Matlab instance'
	
	def __del__(self):
		try:
			self.eng.engClose.argtypes = [c_void_p]
			self.eng.engClose.restype = c_int
			self.eng.engClose(self.matlab)
		except Exception:
			return
	
	"""
	Executes Matlab expression. Returns empty string unless Verbose=True 
	where any standard output from Matlab is returned. 

	expr: String of Matlab expression to be executed
	verbose: Flag to togle capture of standard output from Matlab
	"""	
	def execExpression(self, expr, verbose=False):
		bufLength = 1240
		buf = create_string_buffer('\000', bufLength)
		if verbose==True:
			self.eng.engOutputBuffer.argtypes = [c_void_p, c_void_p, c_int]
			self.eng.engOutputBuffer.restypes = c_int
			self.eng.engOutputBuffer(self.matlab, buf, bufLength)
		
		self.eng.engEvalString.argtypes = [c_void_p, c_char_p]
		self.eng.engEvalString.restype = c_int
		res = self.eng.engEvalString(self.matlab, expr)

		if verbose==True:
			return buf.value
	
	"""
	Gets Matlab variable with the specified name. Returns None if variable doesn't exist
	"""
	def get(self, name):
		self.eng.engGetVariable.argtypes = [c_void_p, c_char_p]
		self.eng.engGetVariable.restype = c_void_p
		obj = self.eng.engGetVariable(self.matlab, name)

		if obj==None:
			raise Exception, 'Variable %s was not found' % name

		val =  self.__getValue(obj)
		
		self.mx.mxDestroyArray.argtypes = [c_void_p]
		self.mx.mxDestroyArray(obj)

		return val
	
	"""
	Returns Python object corresponding to Matlab variable pointer
	"""	
	def __getValue(self, obj):
		if obj==None:
			return None
		elif self.__isNumeric(obj):
			return self.__getNumber(obj)
		elif self.__isString(obj):
			return self.__getString(obj)
		elif self.__isStruct(obj):
			return self.__getStruct(obj)
		elif self.__isCell(obj):
			return self.__getCell(obj)
		else:
			raise Exception, 'Variable type not supported!'
	
	def __getCell(self, obj):
		dim = self.__getDimension(obj)
		
		self.mx.mxGetCell.argtypes = [c_void_p, c_int]
		self.mx.mxGetCell.restype = c_void_p

		res = list()
		for m in range(0,dim[0]):
			for n in range(0, dim[1]):
				item = self.mx.mxGetCell(obj, self.__getIndex(obj, (m,n)))
				item = self.__getValue(item)

				if dim[0]>1:
					res[m].append(item)
				else:
					res.append(item)
		return res

	def __getStruct(self, obj):
		self.mx.mxGetNumberOfFields.argtypes = [c_void_p]
		self.mx.mxGetNumberOfFields.restype = c_int
		fields = self.mx.mxGetNumberOfFields(obj)
		
		res = dict()
		
		self.mx.mxGetFieldNameByNumber.argtypes = [c_void_p, c_int]
		self.mx.mxGetFieldNameByNumber.restype = c_char_p
		
		if self.version == 7:
			self.mx.mxGetFieldByNumber.argtypes = [c_void_p, c_int, c_int]
			self.mx.mxGetFieldByNumber.restype = c_void_p
		else:
			self.mx.mxGetFieldByNumber_730.argtypes = [c_void_p, c_int, c_int]
			self.mx.mxGetFieldByNumber_730.restype = c_void_p
		
		for i in range(0, fields):
			k = self.mx.mxGetFieldNameByNumber(obj, i)
			if self.version==7:
				o = self.mx.mxGetFieldByNumber(obj, 0, i)
			else:
				o = self.mx.mxGetFieldByNumber_730(obj, 0, i)
			res[k] = self.__getValue(o)
		
		return res
	
	def __getNumber(self, obj):
		if self.__isClass(obj, 'double'):
			dtype = numpy.double
		elif self.__isClass(obj, 'int32'):
			dtype = numpy.int32
		else:
			raise Exception, 'Datatype not yet supported!'

		dims = self.__getDimension(obj)
		res = numpy.ndarray(shape=dims, order='fortran', dtype=dtype)
		
		self.__memcpy(self.__getNumpyPointer(res), self.__getMatlabPointer(obj), self.__getNumpySize(res))
		
		return res
		
	def __getString(self, obj):
		dims = self.__getDimension(obj)
		str = create_string_buffer(dims[1])
		
		self.mx.mxGetString.argtypes = [c_void_p, c_char_p, c_int]
		self.mx.mxGetString.restype = c_int
		
		self.mx.mxGetString(obj, str, sizeof(c_int)*dims[1])
		
		return str.value

	"""
	Sends Python variable to the Matlab engine with the specified name
	"""	
	def put(self, name, value):
		var = self.__putValue(value)
			
		self.eng.engPutVariable.argtypes = [c_void_p, c_char_p, c_void_p]
		self.eng.engPutVariable(self.matlab, name, var)
		
		self.mx.mxDestroyArray.argtypes = [c_void_p]
		self.mx.mxDestroyArray(var)

	def __putValue(self, value):
		if type(value)==str:
			return self.__putString(value)
		elif type(value)==dict:
			return self.__putStruct(value)
		elif type(value)==list:
			return self.__putList(value)
		elif type(value)==int or type(value)==float:
			return self.__putScalar(value)
		else:
			return self.__putMatrix(value)
		
			
	def __putScalar(self, value):
		return self.__putMatrix(numpy.array([value]))
		
	def __putString(self, value):
		self.mx.mxCreateString.argtypes = [c_char_p]
		self.mx.mxCreateString.restype = c_void_p
		return self.mx.mxCreateString(c_char_p(value))
		
	def __putMatrix(self, value):
		m = 1 
		n = 1
		
		dtype = value.dtype		
		if value.dtype == numpy.int64:
			dtype = numpy.int32
		elif value.dtype == numpy.float64:
			dtype = numpy.double

		value = numpy.array(value, order='fortran', dtype=dtype)
		
		# Determine size of matrix
		if value.ndim == 2:
			(m, n) = value.shape
		elif value.ndim == 1:
			n = value.shape[0]
		dtype = self.__getClassID(value.dtype)
		
		# Assume matrix is not complex
		isComplex = 0
		
		# Create matlab matrix and copy data
		mat = self.__createNumericMatrix(m, n, dtype, isComplex)
		self.__memcpy(self.__getMatlabPointer(mat), self.__getNumpyPointer(value), self.__getNumpySize(value))
		
		return mat	
	
	def __putStruct(self, value):
		keys = value.keys()
		keys_p = (c_char_p*len(keys))()
		for k in range(0,len(keys)):
			keys_p[k] = c_char_p(keys[k])

		if self.version == 7:
			self.mx.mxCreateStructMatrix.argtypes = [c_int, c_int, c_int, c_void_p]
			self.mx.mxCreateStructMatrix.restype = c_void_p
			self.mx.mxSetFieldByNumber.argtypes = [c_void_p, c_int, c_int, c_void_p]
		else:
			self.mx.mxSetFieldByNumber_730.argtypes = [c_void_p, c_int, c_int, c_void_p]
			self.mx.mxCreateStructMatrix_730.argtypes = [c_int, c_int, c_int, c_void_p]
			self.mx.mxCreateStructMatrix_730.restype = c_void_p
		
		res = 0	
		if self.version == 7:
			res = self.mx.mxCreateStructMatrix(1, 1, len(keys), keys_p)
		else:
			res = self.mx.mxCreateStructMatrix_730(1, 1, len(keys), keys_p)
		
		for k in range(0, len(keys)):
			val = self.__putValue(value[keys[k]])
			if self.version == 7:
				self.mx.mxSetFieldByNumber(res, 0, k, val)
			else:
				self.mx.mxSetFieldByNumber_730(res, 0, k, val)
		return res
	
	def __putList(self, value):
		m = 0
		n = 0
		if len(value)==0:
			pass
		elif type(value[0])==list:
			m = len(value)
			n = len(value[0])	
		else:
			m = 1
			n = len(value)
			value = [value]
		obj = self.mx.mxCreateCellMatrix(m, n)

		self.mx.mxSetCell.argtypes = [c_void_p, c_int, c_void_p]
	
		for row in range(0, m):
			for col in range(0, n):
				id = self.__getIndex(obj, (row, col))
				item = self.__putValue(value[row][col])
				self.mx.mxSetCell(obj, id, item)
		return obj

	def __createNumericMatrix(self, m, n, dtype, isComplex):
		if self.version == 7:
			self.mx.mxCreateNumericMatrix.argtypes = [c_int, c_int, c_int, c_int]
			self.mx.mxCreateNumericMatrix.restype = c_void_p
			return self.mx.mxCreateNumericMatrix(m, n, dtype, isComplex)
		else:
			self.mx.mxCreateNumericMatrix_700.argtypes = [c_int, c_int, c_int, c_int]
			self.mx.mxCreateNumericMatrix_700.restype = c_void_p
			return self.mx.mxCreateNumericMatrix_700(m, n, dtype, isComplex)

	def __setPointer(self, obj, pointer):
		self.mx.mxSetPr.argtypes = [c_void_p, c_void_p]
		self.mx.mxSetPr(obj, pointer)
		
	def __getMatlabPointer(self, obj):
		self.mx.mxGetPr.argtypes = [c_void_p]
		self.mx.mxGetPr.restype = c_void_p
		return self.mx.mxGetPr(obj)
	
	def __getNumpyPointer(self, obj):
		return obj.ctypes.data_as(POINTER(c_void_p))
	
	def __getNumpySize(self, obj):
		return obj.itemsize*obj.size
	
	def __memcpy(self, dst, src, length):
		self.c.memcpy.argtypes = [c_void_p, c_void_p, c_uint]
		self.c.memcpy(dst, src, length)
	
	def __isClass(self, obj, classname):
		self.mx.mxIsClass.argtypes = [c_void_p, c_char_p]
		self.mx.mxIsClass.restype = c_int
		return self.mx.mxIsClass(obj, classname)
	
	def __getClassID(self, dtype):
		if dtype == numpy.double:
			return 6
		elif dtype == numpy.int32:
			return 12
		else:
			raise Exception, 'Matlab data type is not supported'
		
	def __isString(self, obj):
		self.mx.mxIsChar.argtypes = [c_void_p]
		self.mx.mxIsChar.restype = c_int
		return self.mx.mxIsChar(obj)
		
	def __isNumeric(self, obj):
		self.mx.mxIsNumeric.argtypes = [c_void_p]
		self.mx.mxIsNumeric.restype = c_int
		return self.mx.mxIsNumeric(obj)
	
	def __isStruct(self, obj):
		self.mx.mxIsStruct.argtypes = [c_void_p]
		self.mx.mxIsStruct.restype = c_int
		return self.mx.mxIsStruct(obj)
	
	def __isCell(self, obj):
		self.mx.mxIsCell.argtypes = [c_void_p]
		self.mx.mxIsCell.restype = c_int
		return self.mx.mxIsCell(obj)
		
	def __getNumberOfDimensions(self, obj):
		if self.version == 7:
			self.mx.mxGetNumberOfDimensions.argtypes = [c_void_p]
			self.mx.mxGetNumberOfDimensions.restype = c_int
			return self.mx.mxGetNumberOfDimensions(obj)
		else:
			self.mx.mxGetNumberOfDimensions_730.argtypes = [c_void_p]
			self.mx.mxGetNumberOfDimensions_730.restype = c_int
			return self.mx.mxGetNumberOfDimensions_730(obj)
		
	def __getDimension(self, obj):
		self.mx.mxGetM.argtypes = [c_void_p]
		self.mx.mxGetM.restype = c_int
		self.mx.mxGetN.argtypes = [c_void_p]
		self.mx.mxGetN.restype = c_int
		
		return [self.mx.mxGetM(obj), self.mx.mxGetN(obj)]
	
	def __getIndex(self, obj, dim):
		self.mx.mxCalcSingleSubscript.argtypes = [c_void_p, c_int, c_void_p]
		self.mx.mxCalcSingleSubscript.restype = c_int

		d = (c_int*len(dim))()
		for i in range(0,len(dim)):
			d[i] = dim[i]

		return self.mx.mxCalcSingleSubscript(obj, len(dim), d)

	def doCall(self, args, retnum=0, verbose=False):
		if type(args)!=type(list()):
			args = [args]
			
		argCounter = 0
		argNames = []
		for arg in args:
			try:
				if arg.__class__==self.Local:
					argNames.append(arg.name)
				else:
					self.put('arg%03d'%argCounter, arg)
					argNames.append('arg%03d'%argCounter)
			except Exception as e:
				print 'Argument %d raised exception' % argCounter
				raise e
			argCounter = argCounter + 1
		
		returnArguments = string.join(map(lambda r: 'ret%03d'%r, range(0, retnum)))
		functionArguments = string.join(argNames, ',')
		
		if retnum>0:
			command = '[%s] = %s(%s);' % (returnArguments, self.funcName, functionArguments)
		else:
			command = '%s(%s);' % (self.funcName, functionArguments)
		
		res = self.execExpression(command, verbose)

		if verbose==True:
			print res

		if retnum==0:
			return []
			
		returnArgs = []

		for ret in range(0, retnum):
			returnArgs.append(self.get('ret%03d'%ret))
		
		self.clear(returnArguments)

		return returnArgs
	
	"""
	Excecutes matlab function as if it was a Python member function of Matlab. 

	Matlab.X(arguments[, retnum=0[, debug=false]])

	X: Matlab functioname to call (either native or from a M/Mex file)
	arguments: list of arguments to pass to function X
	retnum: Number of expected variables to be returend from X
	debug: Toggle whether to print any standard output data that is generated from Matlab 	
	"""
	def __getattr__(self, a):
		self.funcName = str(a)
		return self.doCall
		
class LocalMatlab(Matlab):
	pass		
		
class ServerMatlab(Matlab):
	pass
	
		

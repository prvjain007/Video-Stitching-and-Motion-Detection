1. Install OpenCV
	1.1 Install OpenCV library in python
	1.2 Add appropriate version of OpenCV bin paths to path.
	    Eg: For Visual Studio 2013 with VC++ 12 and a 64 bit system and 64 bit python
		add ".\build\x64\vc12\bin" to path
2. Check the source of the cameras. Edit the specific camera streams in the code. The program will show runtime error if the streams of the carmeras are out of order
3. Run file "run.bat" to start the code. This code is tested on Python 2.7. But it is expected to work on python 3
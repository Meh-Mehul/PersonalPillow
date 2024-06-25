# PersonalPillow
An image manipulation library in C++. Contains most of the popular filters for image manipulation. It uses the ```stb image``` libraries for loading and writing images.

Sample images and their ouputs to various functions have been placed in ```imgs``` directory.

## Requirements
1. You need to have ```make``` (or mingw32-make if g++ compiler) to run the makefile and build the files (see below for alternative)
2. It is preferred to use 3 channel ```.jpg``` files with this library as i've tested most of the functions on those only. (ig ```.png``` would work maybe just fine).
## Usage
This library provides almost all the famous image filters, which can be applied onto an image of (preferred) 3 channels. All the functions in the library come under the ```Image``` struct.
Sample usage of each function is provided in ```main.cpp``` file. This library should be able to handle image upto the size of approx. ```8000 X 4000```, but may take time in processing.
## Alternative to make
run the command:-

```g++ -o out src/Image.cpp main.cpp```

to get the out.exe file and the execute it by running:-

```./build/out```
## Also:
I planned on making a ```imshow``` function like the one in ```cv2```, but since i used the old graphics.h library i was unable to make it resizable alongwith other problems, so if u want to use the ```imshow_slow``` function:
1. uncomment the function in ```Image.cpp``` file (its at around line 1184)
2. include graphics.h in header file, make sure u have included it in your system too.
3. in the makefile, add the ```CFLAGS2``` after the executing line (line 10).

I may warn you, it not that great, but rest of the functions might work just fine.

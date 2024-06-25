CC			:= g++
BUILDDIR	:= build
SRCDIR		:= src
CFLAGS		:= -std=c++17 -g
CFLAGS2     := -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 

PHONY: make
make:
	@echo Buiding...
	@$(CC) $(CFLAGS) -o $(BUILDDIR)/out $(SRCDIR)/Image.cpp main.cpp 
	@echo yanked out.exe to $(BUILDDIR)
	@echo Running...
	@./$(BUILDDIR)/out.exe
	@echo Compiled and Run Success!
	
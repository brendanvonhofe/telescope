PLUGINS ?= TriMapShift.so
UNAME_S := $(shell uname -s)
CXXFLAGS ?= -std=c++11 -g -c

LIBS ?= -l DDImage

ifeq ($(UNAME_S),Linux)
  NUKE_DIR += /usr/local/Nuke11.3v2
  COMPILER += g++-4.8
  LINK += g++-4.8
  CXXFLAGS += -I $(NUKE_DIR)/include \
	      -fPIC -msse
  LINKFLAGS += -L $(NUKE_DIR) -L ./ -shared
  BINARY += .so
else
  ifeq ($(UNAME_S),Darwin)
    NUKE_DIR += /Applications/Nuke11.2v4/Nuke11.2v4.app/Contents/MacOS
    COMPILER += clang++
    LINK += clang++
    CXXFLAGS += -I $(NUKE_DIR) \
		-F/Applications/Nuke11.2v4/Nuke11.2v4.app/Contents/Frameworks
    LINKFLAGS += -L $(NUKE_DIR) -L ./ -bundle
    LIBS += -framework QuartzCore \
	    -framework IOKit \
	    -framework CoreFoundation \
	    -framework Carbon \
	    -framework ApplicationServices \
	    -framework OpenGL \
	    -framework AGL
    BINARY += .dylib
  endif
endif

all: ndkexists $(PLUGINS)

.PRECIOUS : %.os
%.os: %.cpp
	$(COMPILER) $(CXXFLAGS) -o $(@) $<
%$(BINARY): %.os
	$(LINK) $(LINKFLAGS) $(LIBS) -o $(@) $<
%.a: %.cpp
	$(COMPILER) $(CXXFLAGS) -o lib$(@) $<

ndkexists:
	if test -d $(NUKE_DIR); \
	then echo "using NUKE_DIR from ${NUKE_DIR}"; \
	else echo "NUKE_DIR dir not found! Please set NUKE_DIR"; exit 2; \
	fi
clean:
	rm -rf *.os \
	       *.o \
	       *.a \
	       *.so \
	       *.dylib

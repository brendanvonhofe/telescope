# Compiler Info ('clang++ --version')
# Apple LLVM version 8.0.0 (clang-800.0.42.1)
# Target: x86_64-apple-darwin16.7.0
# Thread model: posix
# InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
# End Compiler Info Output
CXX ?= clang++
LINK ?= clang++
NDKDIR ?= /Applications/Nuke11.2v4/Nuke11.2v4.app/Contents/MacOS
#You will need to set the path to boost for psdReader only
BOOSTDIR ?= REPLACE_ME_WITH_BOOST_DIR
#You will need to set the path to openEXR for exr plugins only
OPENEXRDIR ?= REPLACE_ME_WITH_OPENEXR_DIR
CXXFLAGS ?= -g -c \
            -DUSE_GLEW \
            -I$(NDKDIR)/include \
            -F../../../Nuke11.2v4.app/Contents/Frameworks \
            -DFN_EXAMPLE_PLUGIN 
LINKFLAGS ?= -L$(NDKDIR) \
             -L./ 
LIBS ?= -lDDImage -lGLEW # DDImage and its dependencies
LINKFLAGS += -bundle
FRAMEWORKS ?= -framework QuartzCore \
              -framework IOKit \
              -framework CoreFoundation \
              -framework Carbon \
              -framework ApplicationServices \
              -framework OpenGL \
              -framework AGL 
all: ndkexists TriMapShift.dylib

exrPlugins: openEXRexists exrReader.so exrWriter.so

exrReader.dylib: exrReader.os ExrChannelNameToNuke.os
	$(LINK) $(LINKFLAGS) \
	        -L$(OPENEXRDIR)/lib \
	        -I$(OPENEXRDIR)/include \
	        $(LIBS) \
	        -lIlmImf-2_2_Foundry \
	        -lHalf-2_2_Foundry \
	        -lIex-2_2_Foundry \
	        -lIexMath-2_2_Foundry \
	        -lIlmThread-2_2_Foundry \
	        -lImath-2_2_Foundry \
	        $(FRAMEWORKS) \
	        -o $(@) $^

exrWriter.dylib: exrWriter.os ExrChannelNameToNuke.os
	$(LINK) $(LINKFLAGS) \
	        -L$(OPENEXRDIR)/lib \
	        -I$(OPENEXRDIR)/include \
	        $(LIBS) \
	        -lIlmImf-2_2_Foundry \
	        -lHalf-2_2_Foundry \
	        -lIex-2_2_Foundry \
	        -lIexMath-2_2_Foundry \
	        -lIlmThread-2_2_Foundry \
	        -lImath-2_2_Foundry \
	        $(FRAMEWORKS) \
	        -o $(@) $^

exrReader.os: exrReader.cpp
	$(CXX) $(CXXFLAGS) \
	       -I$(OPENEXRDIR)/include \
	       -o $(@) $<

exrWriter.os: exrWriter.cpp
	$(CXX) $(CXXFLAGS) \
	       -I$(OPENEXRDIR)/include \
	       -o $(@) $<

ExrChannelNameToNuke.os: ExrChannelNameToNuke.cpp
	$(CXX) $(CXXFLAGS) \
	       -I$(OPENEXRDIR)/include \
	       -o $(@) $<

openEXRexists:
	if test -d $(OPENEXRDIR); \
	then echo "using OPENEXRDIR from ${OPENEXRDIR}"; \
	else echo "OPENEXRDIR dir not found! Please set OPENEXRDIR"; exit 2; \
	fi

psdReader: psdReader.dylib

psdReader.dylib: psdReader.os
	$(LINK) $(LINKFLAGS) \
	        -L$(OPENEXRDIR)/lib \
	        -I$(OPENEXRDIR)/include \
	        $(LIBS) \
	        -lIlmImf-2_2_Foundry \
	        $(FRAMEWORKS) \
	        -o $(@) $^

psdReader.os: psdReader.cpp boostexists
	$(CXX) $(CXXFLAGS) \
	       -I$(BOOSTDIR)/include \
	       -o $(@) $<

boostexists:
	if test -d ${BOOSTDIR}; \
	then echo "using boost from ${BOOSTDIR}"; \
	else echo "boost dir not found! Please set BOOSTDIR"; exit 2; \
	fi

.PRECIOUS : %.os
%.os: %.cpp
	$(CXX) $(CXXFLAGS) -o $(@) $<
%.dylib: %.os
	$(LINK) $(LINKFLAGS) $(LIBS) $(FRAMEWORKS) -o $(@) $<
%.a: %.cpp
	$(CXX) $(CXXFLAGS) -o lib$(@) $<

ndkexists:
	if test -d $(NDKDIR); \
	then echo "using NDKDIR from ${NDKDIR}"; \
	else echo "NDKDIR dir not found! Please set NDKDIR"; exit 2; \
	fi
clean:
	rm -rf *.os \
	       *.o \
	       *.a \
	       *.dylib
